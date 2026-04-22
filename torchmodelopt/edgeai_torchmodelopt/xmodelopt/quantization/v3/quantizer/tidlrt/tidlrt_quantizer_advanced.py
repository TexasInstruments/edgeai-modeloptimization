
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


import copy
from typing import Callable, Optional, List
import itertools
import types
import torch
import torch.nn.functional as F
from torch.fx import Node
# from torch.ao.quantization.pt2e.utils import (
from torchao.quantization.pt2e.utils import (
    _get_aten_graph_module_for_pattern,
    _is_conv_node,
    _is_conv_transpose_node,
)
# from torch.ao.quantization.quantizer.utils import (
from torchao.quantization.pt2e.quantizer.utils import (
    annotate_input_qspec_map as _annotate_input_qspec_map,
    annotate_output_qspec as  _annotate_output_qspec,
)
# from torch.ao.quantization.quantizer import (
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec,
    QuantizationSpecBase
)
# from torch.ao.quantization.pt2e.export_utils import _WrapperModule
from torchao.quantization.pt2e.export_utils import WrapperModule as _WrapperModule
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


from ..xnnpack import XNNPACKQuantizer, QuantizationConfig
from ..xnnpack import register_annotator, AnnotatorType
from ..xnnpack import get_input_act_qspec, get_output_act_qspec, get_weight_qspec, get_bias_qspec
from ..xnnpack import _is_annotated, _mark_nodes_as_annotated, get_annotation_func, _is_input_large_scalar, _is_input_non_float_tensor
from ..xnnpack import OP_TO_ANNOTATOR


def get_aten_overload_ops(aten_op_name: str):
    aten_op_overload_packet = getattr(torch.ops.aten, aten_op_name)
    overload_names = aten_op_overload_packet.overloads()
    op_overloads = []
    for overload_name in overload_names:
        op_overload = getattr(aten_op_overload_packet, overload_name)
        op_overloads += [op_overload]
    #
    return op_overloads


def _do_annotate_conv_mul_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]],
    has_relu: bool,
    is_conv_transpose: bool = False,
) -> list[list[Node]]:
    """
    Given a function that takes in a `conv_fn` and returns a conv-bn[-relu] pattern,
    return a list of annotated partitions.

    The output of the pattern must include a dictionary from string name to node
    for the following names: "input", "conv", "weight", "bias", and "output".
    """

    # Example inputs for conv-bn1d patterns
    _conv1d_mul_add_example_inputs = (
        torch.randn(1, 1, 3),  # x
        torch.randn(1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # mul
        torch.randn(1),  # add
    )

    # Example inputs for conv-bn2d patterns
    _conv2d_mul_add_example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # mul
        torch.randn(1),  # add
    )

    def get_pattern(conv_fn: Callable, relu_is_inplace: bool):
        def _conv_mul_add(x, conv_weight, conv_bias, mul_weight, add_weight):
            conv_out = conv_fn(x, conv_weight, conv_bias)
            mul_out = conv_out.mul(mul_weight)
            add_out = mul_out.add(add_weight)
            if has_relu:
                output = F.relu_(add_out) if relu_is_inplace else F.relu(add_out)
            else:
                output = add_out
            return output, {
                "input": x,
                "conv": conv_out,
                "weight": conv_weight,
                "bias": conv_bias,
                'mul': mul_out,
                'add': add_out,
                "output": output,
            }

        return _WrapperModule(_conv_mul_add)

    # Needed for matching, otherwise the matches gets filtered out due to unused
    # nodes returned by batch norm
    gm.graph.eliminate_dead_code()
    gm.recompile()

    matches = []
    if is_conv_transpose:
        combinations = [
            (F.conv_transpose1d, _conv1d_mul_add_example_inputs),
            (F.conv_transpose2d, _conv2d_mul_add_example_inputs),
        ]
    else:
        combinations = [
            (F.conv1d, _conv1d_mul_add_example_inputs),  # type: ignore[list-item]
            (F.conv2d, _conv2d_mul_add_example_inputs),  # type: ignore[list-item]
        ]

    # Add `is_cuda` and `relu_is_inplace` dimensions
    combinations = itertools.product(  # type: ignore[assignment]
        combinations,
        [True, False] if torch.cuda.is_available() else [False],  # is_cuda
        [True, False] if has_relu else [False],  # relu_is_inplace
    )

    # Match against all conv dimensions and cuda variants
    for (conv_fn, example_inputs), is_cuda, relu_is_inplace in combinations:  # type: ignore[misc]
        pattern = get_pattern(conv_fn, relu_is_inplace)  # type: ignore[has-type]
        pattern = _get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda)  # type: ignore[has-type]
        pattern.graph.eliminate_dead_code()
        pattern.recompile()
        matcher = SubgraphMatcherWithNameNodeMap(pattern, ignore_literals=True)
        matches.extend(matcher.match(gm.graph))

    # Annotate nodes returned in the matches
    annotated_partitions = []
    for match in matches:
        name_node_map = match.name_node_map
        input_node = name_node_map["input"]
        conv_node = name_node_map["conv"]
        weight_node = name_node_map["weight"]
        bias_node = name_node_map["bias"]
        mul_node = name_node_map["mul"]
        add_node = name_node_map["add"]
        output_node = name_node_map["output"]

        # TODO: annotate the uses of input, weight, and bias separately instead
        # of assuming they come from a single conv node. This is not possible today
        # because input may have multiple users, and we can't rely on the conv node
        # always being the first user. This was the case in models with skip
        # connections like resnet18

        # Validate conv args
        if conv_node.args[0] is not input_node:
            raise ValueError("Conv arg did not contain input node ", input_node)
        if conv_node.args[1] is not weight_node:
            raise ValueError("Conv arg did not contain weight node ", weight_node)
        if len(conv_node.args) > 2 and conv_node.args[2] is not bias_node:
            raise ValueError("Conv arg did not contain bias node ", bias_node)

        # Skip if the partition is already annotated or is filtered out by the user
        partition = [add_node, mul_node, conv_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)

        if _is_annotated(partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        # Annotate conv inputs and pattern output
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        if bias_node is not None:
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        for node, qspec in input_qspec_map.items():
            _annotate_input_qspec_map(conv_node, node, qspec)
        
        _annotate_output_qspec(output_node, get_output_act_qspec(quantization_config))
        
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("conv_mul_add_relu")
def _annotate_conv_mul_add_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv_transpose + batchnorm + relu partitions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_mul_add(
        gm, quantization_config, filter_fn, has_relu=True, is_conv_transpose=False
    )


@register_annotator("conv_mul_add")
def _annotate_conv_mul_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    """
    Find conv_transpose + batchnorm + relu partitions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    return _do_annotate_conv_mul_add(
        gm, quantization_config, filter_fn, has_relu=False, is_conv_transpose=False
    )


@register_annotator("mul_add")
def _annotate_mul_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
        ]:
            continue
        add_node = node
        maybe_mul = node.args[0]
        if (
            not isinstance(maybe_mul, Node)
            or maybe_mul.op != "call_function"
            or maybe_mul.target
            not in [
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.mul_.Tensor,
            ]
        ):
            continue

        mul_node = maybe_mul
        if len(mul_node.users) > 1:
            # mul can't be fused with ReLU if the result of mul is being used
            # else where in the graph
            continue

        partition = [add_node, mul_node]

        if _is_annotated(partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            partition.append(input_act0)
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            partition.append(input_act1)
            input_qspec_map[input_act1] = input_act_qspec
        
        for node, qspec in input_qspec_map.items():
            _annotate_input_qspec_map(mul_node, node, qspec)
        
        _annotate_output_qspec(add_node, output_act_qspec)
        annotated_partitions.append(partition)
    
    return annotated_partitions


def _do_annotate_linear_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
    has_relu: bool = False
) -> Optional[list[list[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
            continue
        add_node = node

        maybe_linear = None
        for node_l in add_node.args:
            if (isinstance(node_l, Node) and node_l.op == "call_function" and node_l.target in [torch.ops.aten.linear.default]):
                maybe_linear = node_l
            
        if not maybe_linear or len(maybe_linear.users) > 1:
            # linear can't be fused with add if the result of linear is being used
            # else where in the graph
            continue
        
        linear_node = maybe_linear
        partition = [add_node, linear_node]
        output_node = add_node

        if has_relu and len(add_node.users) == 1:
            # why is the users value None although key is valid? - use .next anyway for now
            maybe_relu = list(add_node.users.keys())[0]
            if (isinstance(maybe_relu, Node) and maybe_relu.op == "call_function"
                and maybe_relu.target in [torch.ops.aten.relu.default, torch.ops.aten.relu_.default]):
                output_node = maybe_relu
                partition = [output_node] + partition

        if _is_annotated(partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        input_weight_qspec = get_weight_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = linear_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            # partition.append(input_act0)
            input_qspec_map[input_act0] = input_act_qspec

        input_weight = linear_node.args[1]
        if isinstance(input_weight, Node):
            if _is_input_large_scalar(input_weight, gm):
                continue
            if _is_input_non_float_tensor(input_weight):
                continue
            partition.append(input_weight)
            input_qspec_map[input_weight] = input_weight_qspec

        _mark_nodes_as_annotated(partition)
        for node, qspec in input_qspec_map.items():
            _annotate_input_qspec_map(linear_node, node, qspec)
        
        _annotate_output_qspec(output_node, output_act_qspec)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("linear_add")
def _annotate_linear_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None
) -> Optional[list[list[Node]]]:
    return _do_annotate_linear_add(gm, quantization_config, filter_fn, has_relu=False)


@register_annotator("linear_add_relu")
def _annotate_linear_add_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None
) -> Optional[list[list[Node]]]:
    return _do_annotate_linear_add(gm, quantization_config, filter_fn, has_relu=True)


def _do_annotate_matmul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
    has_add: bool = False,
    has_relu = False
) -> Optional[list[list[Node]]]:
    # matmul is currently not quantized
    # quantizing it needs carefull handling of attention layers
    annotated_partitions = []
    is_node_variable_dict = gm._is_node_variable_dict
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.matmul.default:
            continue
        # if filter_fn and not filter_fn(node):
        #     continue

        matmul_node = node
        nodes_to_mark_annotated = [matmul_node]
        
        input_act_qspec = get_input_act_qspec(quantization_config)
        input_weight_qspec = get_weight_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        
        input1, input2 = matmul_node.args
        
        var1 = isinstance(input1, Node) and is_node_variable_dict[input1]
        var2 = isinstance(input2, Node) and is_node_variable_dict[input2]
        
        input_type_qscheme_map = {
            (True,True)  : (torch.per_tensor_symmetric, torch.per_tensor_symmetric,  torch.per_tensor_symmetric),
            (True,False) : (torch.per_tensor_symmetric, torch.per_channel_symmetric, torch.per_tensor_symmetric),
            (False,True) : (torch.per_tensor_symmetric, torch.per_tensor_symmetric,  torch.per_tensor_symmetric),
            (False,False): (torch.per_tensor_symmetric, torch.per_channel_symmetric, torch.per_tensor_symmetric),
        }
        
        from ...qconfig_types import create_qspec_deepcopy_with_kwargs
        specs = [input_act_qspec, input_weight_qspec, output_act_qspec]
        for i, spec in enumerate(specs):
            if spec is None:
                continue
            if spec.qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
                continue
            if spec.qscheme == torch.per_tensor_affine:
                specs[i] = create_qspec_deepcopy_with_kwargs(spec, quantization_config.is_qat, qscheme=torch.per_tensor_symmetric)
            if spec.qscheme == torch.per_channel_affine:
                specs[i] = create_qspec_deepcopy_with_kwargs(spec, quantization_config.is_qat, qscheme=torch.per_channel_symmetric)

        input_act_qspec, input_weight_qspec, output_act_qspec = specs
        input_qspec_map = {}
        
        # input_act_qspec.qscheme
        # if var2 is True:
            # continue
        if isinstance(input1, Node):
            nodes_to_mark_annotated.append(input1)
            input_qspec_map[input1] = input_act_qspec
        
        if isinstance(input2, Node):
            nodes_to_mark_annotated.append(input2)
            input_qspec_map[input2] = input_act_qspec
        
        current_node = matmul_node
        out_node = None
        if has_add and len(current_node.users) == 1:
            next_node = list(current_node.users.keys())[0]
            if next_node.op == "call_function" and next_node.target in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
                out_node = next_node
                nodes_to_mark_annotated = [out_node] + nodes_to_mark_annotated
                current_node = out_node
        if has_relu and len(current_node.users)==1:
            next_node = list(current_node.users.keys())[0]
            if next_node.op == "call_function" and next_node.target in [torch.ops.aten.relu.default, torch.ops.aten.relu_.default]:
                out_node = next_node
                nodes_to_mark_annotated = [out_node] + nodes_to_mark_annotated
        
        if _is_annotated(nodes_to_mark_annotated):
            continue
        if filter_fn and any(not filter_fn(n) for n in nodes_to_mark_annotated):
            continue
    
        for node, qspec in input_qspec_map.items():
            _annotate_input_qspec_map(matmul_node, node, qspec)
        
        _annotate_output_qspec(out_node or matmul_node, output_act_qspec)
        
        # if out_node:
        #     if 'quantization_annotation' not in out_node.meta or not out_node.meta["quantization_annotation"]._annotated :
        #         out_node.meta["quantization_annotation"] = QuantizationAnnotation(
        #                 # input_qspec_map=input_qspec_map,
        #                 output_qspec=output_act_qspec,
        #                 _annotated=True,
        #             )
        #     matmul_node.meta["quantization_annotation"] = QuantizationAnnotation(
        #             input_qspec_map=input_qspec_map,
        #             # output_qspec=output_act_qspec,
        #             _annotated=True,
        #         )
        # else:
        #     matmul_node.meta["quantization_annotation"] = QuantizationAnnotation(
        #             input_qspec_map=input_qspec_map,
        #             output_qspec=output_act_qspec,
        #             _annotated=True,
        #         )
    
        _mark_nodes_as_annotated(nodes_to_mark_annotated)
        annotated_partitions.append(nodes_to_mark_annotated)

    return annotated_partitions


@register_annotator('matmul')
def annotate_matmul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
    ):
    return _do_annotate_matmul(gm, quantization_config, filter_fn,)


@register_annotator('matmul_add')
def annotate_matmul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
    ):
    return _do_annotate_matmul(gm, quantization_config, filter_fn, has_add=True)


@register_annotator('matmul_relu')
def annotate_matmul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
    ):
    return _do_annotate_matmul(gm, quantization_config, filter_fn, has_relu=True)


@register_annotator('matmul_add_relu')
def annotate_matmul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
    ):
    return _do_annotate_matmul(gm, quantization_config, filter_fn, has_add=True, has_relu=True)


@register_annotator('layernorm')
def annotate_layernorm(    
        gm: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ):
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)
    weight_qspec = get_weight_qspec(quantization_config)
    bias_qspec = get_bias_qspec(quantization_config)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.layer_norm.default:
            continue
        if filter_fn and not filter_fn(node):
            continue
        act_node = node.args[0]
        weight_node = node.args[2]
        bias_node = None
        if len(node.args) > 3:
            bias_node = node.args[3]

        if _is_annotated([node]) is False:  # type: ignore[list-item]
            _annotate_input_qspec_map(
                node,
                act_node,
                input_act_qspec,
            )
            _annotate_input_qspec_map(
                node,
                weight_node,
                weight_qspec,
            )
            nodes_to_mark_annotated = [node, weight_node]
            if bias_node:
                _annotate_input_qspec_map(
                    node,
                    bias_node,
                    bias_qspec,
                )
                nodes_to_mark_annotated.append(bias_node)
            _annotate_output_qspec(node, output_act_qspec)
            _mark_nodes_as_annotated(nodes_to_mark_annotated)
            annotated_partitions.append(nodes_to_mark_annotated)

    return annotated_partitions

def _prepend_list(orig_list, new_list, prepend=True):
    if prepend:
        orig_list[:0] = new_list
    else:
        orig_list.extend(new_list)
    #
    return orig_list

def get_qspec_attr_list(qspec:QuantizationSpecBase):
    attrs = []
    for d in dir(qspec): 
        if d.startswith('_') :
            continue
        attr = getattr(qspec, d)
        if  isinstance(attr, (types.MethodType, types.FunctionType, type, Callable)):
            continue
        
        attrs.append(d)
    return attrs


class TIDLRTQuantizerAdvanced(XNNPACKQuantizer):
    NEW_ANNOTATION_STATIC_PATTERNS = ['conv_mul_add_relu', 'conv_mul_add', 'linear_add_relu', 'linear_add', 'mul_add', 'matmul', 'matmul_add', 'matmul_relu', 'matmul_add_relu', 'layernorm']
    NEW_ANNOTATION_DYNAMIC_PATTERNS = []
    _prepend_list(XNNPACKQuantizer.STATIC_OPS, NEW_ANNOTATION_STATIC_PATTERNS)
    _prepend_list(XNNPACKQuantizer.DYNAMIC_OPS, NEW_ANNOTATION_DYNAMIC_PATTERNS)

    def __init__(self, *args, annotation_patterns=None, **kwargs):
        super().__init__(*args, **kwargs)
        # select annotators based on annotation_patterns
        if annotation_patterns is None:
            self.STATIC_QAT_ONLY_OPS = copy.deepcopy(self.__class__.STATIC_QAT_ONLY_OPS)
            self.STATIC_OPS = copy.deepcopy(self.__class__.STATIC_OPS)
            self.DYNAMIC_OPS = copy.deepcopy(self.__class__.DYNAMIC_OPS)
        else:
            self.STATIC_QAT_ONLY_OPS = []
            self.STATIC_OPS = []
            self.DYNAMIC_OPS = []
            for n in annotation_patterns:
                if n in OP_TO_ANNOTATOR:
                    if n in self.__class__.DYNAMIC_OPS:
                        self.DYNAMIC_OPS += [n]
                    elif n in self.__class__.STATIC_QAT_ONLY_OPS:
                        self.STATIC_QAT_ONLY_OPS += [n]
                    elif n in self.__class__.STATIC_OPS:
                        self.STATIC_OPS += [n]
                    else:
                        print(f"WARNING: Annotation pattern {n} is unknown - must be one of the following:")
                        print(f"WARNING: DYNAMIC_OPS: {self.DYNAMIC_OPS}  STATIC_QAT_ONLY_OPS: {self.STATIC_QAT_ONLY_OPS} OR STATIC_OPS: {self.STATIC_OPS}") 
                    #
                else:
                    print(f"WARNING: Annotation pattern {n} not one of OP_TO_ANNOTATOR: {OP_TO_ANNOTATOR.keys()}")
                #
            #
        #
        self.STATIC_OPS += self.DYNAMIC_OPS
        pass

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Transforms scalar values to tensor attributes"""
        # there is a bug in this transformation in XNNPACKQuantizer - it is not respecting dtype
        # return _convert_scalars_to_attrs(model)
        return model
    
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        model = super().annotate(model)
        return model


def get_quantizer(is_qat=True, fast_mode=False, device=None, annotation_patterns=None, **kwargs):
    return TIDLRTQuantizerAdvanced(annotation_patterns=annotation_patterns, **kwargs)