#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################

import os
import copy
import torch
from torch.ao.quantization import quantize_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization import FakeQuantize
import statistics

from .... import xnn

from . import observer_types
from . import fake_quanitze_types
from . import qconfig_types


class ModelQuantFormat:
    FLOAT_MODEL = "FLOAT_MODEL"
    FAKEQ_MODEL = "FAKEQ_MODEL"
    QDQ_MODEL = "QDQ_MODEL"
    INT_MODEL = "INT_MODEL"
    _NUM_FORMATS_ = 4


class QuantFxBaseModule(torch.nn.Module):
    def __init__(self, model, *args, qconfig_type=None, example_inputs=None, is_qat=True, backend="qnnpack",
                 total_epochs=0, num_batch_norm_update_epochs=None, num_observer_update_epochs=None,
                 qconfig_mode=qconfig_types.QConfigMode.DEFAULT):
        '''
        model: input model to be used for QAT
        qconfig_type: qconfig_type can be one of the modes defined in qconfig_types (string)
            or it can be a dict that will be passed to qconfig_types.get_config_from_dict()
            it can also be an instance of torch.ao.quantization.QConfig as used when using torch.ao.quantization apis
        '''
        super().__init__()
        if not total_epochs:
            raise RuntimeError("total_epochs must be provided")
        #
        # handle None here
        qconfig_type = qconfig_type or qconfig_types.QConfigType.DEFAULT
        # split based on + for mixed precision
        qconfig_type = qconfig_type.split("+") if isinstance(qconfig_type, str) else (qconfig_type, )
        if len(qconfig_type) > 2:
            raise RuntimeError(f"maximum of 2 entries are supported in qconfig_type:{qconfig_type}")
        #

        qconfig_mapping = qconfig_types.get_qconfig_mapping(is_qat, backend, qconfig_type)
        if is_qat:
            model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
        else:
            model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
        #
        model = qconfig_types.adjust_mixed_precision_qconfig(model, is_qat, backend, qconfig_type)
        self.module = model

        # other parameters
        self.is_qat = is_qat
        self.backend = backend
        self.qconfig_type = qconfig_type
        self.num_batch_norm_update_epochs = num_batch_norm_update_epochs
        self.num_observer_update_epochs = num_observer_update_epochs
        self.num_epochs_tracked = 0
        self.total_epochs = total_epochs
        # set the quantization backend - qnnpack, fbgemm, x86, onednn etc.
        self.set_quant_backend(backend)
        # related to adaptive quantization

        self.qconfig_mode = qconfig_types.QConfigMode(qconfig_mode)
        self.forzen_layer_names_list = []

    def set_quant_backend(self, backend=None):
        if backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError("Quantized backend not supported: " + str(backend))
        torch.backends.quantized.engine = backend

    def load_weights(self, pretrained, *args, strict=True, state_dict_name=None, **kwargs):
        data_dict = torch.load(self, pretrained, *args, **kwargs)
        if state_dict_name:
            state_dict_names = state_dict_name if isinstance(state_dict_name, (list,tuple)) else [state_dict_name]
            for s_name in state_dict_names:
                data_dict = data_dict[s_name] if ((data_dict is not None) and s_name in data_dict) else data_dict
            #
        #
        self.load_state_dict(data_dict, strict=strict)

    def train(self, mode: bool = True):
        # put the model in expected mode
        super().train(mode=mode)
        # also freeze the params if required
        if mode is True:
            # set the default epoch at which freeze occurs during training (if missing)
            num_batch_norm_update_epochs = self.num_batch_norm_update_epochs or ((self.total_epochs//2)-1)
            num_observer_update_epochs = self.num_observer_update_epochs or ((self.total_epochs//2)+1)
            freeze_bn = (self.num_epochs_tracked >= num_batch_norm_update_epochs)
            freeze_observers = (self.num_epochs_tracked >= num_observer_update_epochs)
            if freeze_bn:
                xnn.utils.print_once('Freezing BN for subsequent epochs')
            #
            if freeze_observers:
                xnn.utils.print_once('Freezing ranges for subsequent epochs')
            #
            self.freeze(freeze_bn=freeze_bn, freeze_observers=freeze_observers)
            self.adjust_gradual_quantization()
            self.num_epochs_tracked += 1
        else:
            self.freeze()
        #
        return self

    def adjust_gradual_quantization(self):
        '''
        adjust quantization parameters on epoch basis
        '''
        if self.qconfig_mode != qconfig_types.QConfigMode.DEFAULT and self.total_epochs >= 10:
            # find unstable layers and freeze them
            self.adaptive_freeze_layers(fake_quanitze_types.ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES)

    def is_fake_quant_with_param(self, pmodule, cmodule, fake_quant_types):
        num_params = len(list(pmodule.parameters(recurse=False)))
        return isinstance(cmodule, fake_quant_types) and num_params>0

    def adaptive_freeze_layers(self, fake_quant_types, **kwargs):
        epoch_gradual_quant_start = max(self.total_epochs//2, 1)
        if self.qconfig_mode == qconfig_types.QConfigMode.FREEZE_DEPTHWISE_LAYERS:
            num_total_layers = 0
            self.forzen_layer_names_list = []
            is_freezing_epoch = (self.num_epochs_tracked >= epoch_gradual_quant_start)
            for pname, pmodule in list(self.named_modules()):
                is_input_conv_module = False
                is_depthwise_conv_module = False
                if isinstance(pmodule, torch.nn.Conv2d) and pmodule.in_channels < 8:
                    # too less input channels, could be first conv module
                    is_input_conv_module = True
                if isinstance(pmodule, torch.nn.Conv2d) and pmodule.groups == pmodule.in_channels:
                    is_depthwise_conv_module = True
                #
                for cname, cmodule in list(pmodule.named_children()):
                    if self.is_fake_quant_with_param(pmodule, cmodule, fake_quant_types):
                        is_frozen_layer = (is_input_conv_module or is_depthwise_conv_module)
                        if is_freezing_epoch and is_frozen_layer:
                            # stop updating quantization ranges and stats
                            pmodule.apply(torch.ao.quantization.disable_observer)
                            pmodule.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                            # stop updating parmeters
                            for param in pmodule.parameters(recurse=False):
                                param.requires_update = False
                            #
                            self.forzen_layer_names_list.append(pname)
                        #
                        num_total_layers += 1
                    #
                #
            #
            print(f"using adaptive quantization - qconfig_mode:{self.qconfig_mode} "
                  f"num_frozen_layers:{len(self.forzen_layer_names_list)}/{num_total_layers} ")
        elif self.qconfig_mode == qconfig_types.QConfigMode.FREEZE_UNSTABLE_LAYERS:
            num_total_layers = 0
            delta_change_list = []
            for pname, pmodule in list(self.named_modules()):
                for cname, cmodule in list(pmodule.named_children()):
                    if self.is_fake_quant_with_param(pmodule, cmodule, fake_quant_types):
                        cmodule.set_adaptive_params(detect_change=True, **kwargs)
                        delta_change_list.append(cmodule.delta_change)
                    #
                #
            #
            if self.num_epochs_tracked >= epoch_gradual_quant_start:
                is_freezing_start_epoch = (self.num_epochs_tracked == epoch_gradual_quant_start)
                # find sign_change_threshold
                freeze_fraction = 0.15
                delta_change_min = 0.04
                topk_index = int((len(delta_change_list)-1) * (1-freeze_fraction))
                delta_change_knee = sorted(delta_change_list)[topk_index]
                delta_change_threshold = max(delta_change_knee, delta_change_min)

                # freeze layers with high sign change
                num_total_layers = 0
                for pname, pmodule in list(self.named_modules()):
                    max_delta_change = 0.0
                    for cname, cmodule in list(pmodule.named_children()):
                        if self.is_fake_quant_with_param(pmodule, cmodule, fake_quant_types):
                            # once frozen, always frozen
                            is_frozen_layer = (pname in self.forzen_layer_names_list)
                            is_high_change = is_freezing_start_epoch and (cmodule.delta_change >= delta_change_threshold)
                            if is_frozen_layer or is_high_change:
                                # stop updating delta_change
                                cmodule.set_adaptive_params(detect_change=False, **kwargs)
                                # stop updating quantization ranges and stats
                                pmodule.apply(torch.ao.quantization.disable_observer)
                                pmodule.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                                # stop updating parmeters
                                for param in pmodule.parameters(recurse=False):
                                    param.requires_update = False
                                #
                                self.forzen_layer_names_list.append(pname)
                            #
                            num_total_layers += 1
                        #
                    #
                #
            #
            self.forzen_layer_names_list = list(set(self.forzen_layer_names_list))
            print(f"using adaptive quantization - qconfig_mode:{self.qconfig_mode} "
                  f"median_delta_change:{statistics.median(delta_change_list):.4f} max_delta_change:{max(delta_change_list):.4f} "
                  f"num_frozen_layers:{len(self.forzen_layer_names_list)}/{num_total_layers} "
                  f"frozen_layers:{self.forzen_layer_names_list} ")
        #

    def freeze(self, freeze_bn=True, freeze_observers=True):
        if freeze_observers is True:
            self.apply(torch.ao.quantization.disable_observer)
        elif freeze_observers is False:
            self.apply(torch.ao.quantization.enable_observer)
        #
        if freeze_bn is True:
            self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        elif freeze_bn is False:
            self.apply(torch.nn.intrinsic.qat.update_bn_stats)
        #
        return self

    def unfreeze(self, freeze_bn=False, freeze_observers=False):
        self.freeze(freeze_bn, freeze_observers)
        return self

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)

    def convert(self, device='cpu', model_quant_format=None, convert_custom_config=None, backend_config=None):
        self.freeze()
        # convert requires cpu model
        self.to(torch.device(device))
        # now do the actual conversion
        self.module = quantize_fx.convert_fx(self.module, convert_custom_config=convert_custom_config, backend_config=backend_config)
        return self

    def export(self, example_input, filename='model.onnx', opset_version=17, model_quant_format=None, preserve_qdq_model=True,
               simplify=False, skipped_optimizers=None):
        if model_quant_format == ModelQuantFormat.INT_MODEL:
            # # Convert QDQ format to Int8 format
            import onnxruntime as ort
            qdq_filename = os.path.splitext(filename)[0] + '_qdq.onnx'
            torch.onnx.export(self, example_input, qdq_filename, opset_version=opset_version)
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            so.optimized_model_filepath = filename
            # logger.info("Inplace conversion of QDQ model to INT8 model at: {}".format(onnx_file))
            ort.InferenceSession(qdq_filename, so)
            if not preserve_qdq_model:
                os.remove(qdq_filename)
            #
        else:
            torch.onnx.export(self, example_input, filename, opset_version=opset_version)
        #
        if simplify:
            import onnx
            from onnxsim import simplify
            onnx_model = onnx.load(filename)
            onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizers)
            onnx.save(onnx_model, filename)
        #