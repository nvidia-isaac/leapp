#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Replace TensorRT-unsupported ONNX ops with supported operators.

Only operators that have a local, TensorRT-legal decomposition are rewritten.
String, sequence, and random ops are left in place.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from isaac_deploy_trt.surgery import GraphLike, NodeLike, TensorLike, _gs

logger = logging.getLogger(__name__)

_FLOAT = 1
_UINT8 = 2
_INT8 = 3
_INT32 = 6
_INT64 = 7
_BOOL = 9

_NP_TO_ONNX = {
    np.dtype(np.float32): _FLOAT,
    np.dtype(np.float16): 10,
    np.dtype(np.uint8): _UINT8,
    np.dtype(np.int8): _INT8,
    np.dtype(np.int32): _INT32,
    np.dtype(np.int64): _INT64,
    np.dtype(np.bool_): _BOOL,
}

_BOOL_BITWISE = {
    "BitwiseAnd": "And",
    "BitwiseOr": "Or",
    "BitwiseXor": "Xor",
    "BitwiseNot": "Not",
}

_CONV_ATTRS = ("auto_pad", "dilations", "group", "kernel_shape", "pads", "strides")


def _np_dtype(dtype):
    if dtype is None:
        return None
    try:
        return np.dtype(dtype)
    except TypeError:
        return None


def _onnx_code(dtype) -> Optional[int]:
    np_dtype = _np_dtype(dtype)
    if np_dtype is None:
        return None
    return _NP_TO_ONNX.get(np_dtype)


def _is_bool(dtype) -> bool:
    return _np_dtype(dtype) == np.dtype(np.bool_)


def _node_name(node: NodeLike, fallback: str) -> str:
    return node.name or fallback


def _optional_input(node: NodeLike, index: int) -> Optional[TensorLike]:
    if index >= len(node.inputs):
        return None
    tensor = node.inputs[index]
    if tensor is None or not getattr(tensor, "name", ""):
        return None
    return tensor


def _detach(graph: GraphLike, node: NodeLike) -> None:
    for tensor in list(node.inputs):
        if tensor is None:
            continue
        consumers = getattr(tensor, "outputs", None)
        if consumers is not None and node in consumers:
            consumers.remove(node)
    for tensor in list(node.outputs):
        if tensor is not None:
            tensor.inputs.clear()
    node.inputs.clear()
    node.outputs.clear()
    graph.nodes.remove(node)


def _const(name: str, value, dtype) -> TensorLike:
    gs = _gs()
    return gs.Constant(name=name, values=np.array(value, dtype=dtype))


def _var(name: str, dtype) -> TensorLike:
    gs = _gs()
    return gs.Variable(name=name, dtype=dtype)


def _add(graph: GraphLike, op: str, inputs: Sequence[TensorLike], output: TensorLike, *, name: str, attrs=None) -> NodeLike:
    gs = _gs()
    node = gs.Node(
        op=op,
        name=name,
        attrs=dict(attrs or {}),
        inputs=list(inputs),
        outputs=[output],
    )
    graph.nodes.append(node)
    return node


def _cast(graph: GraphLike, tensor: TensorLike, onnx_type: int, np_dtype, name: str) -> TensorLike:
    if _onnx_code(tensor.dtype) == onnx_type:
        return tensor
    casted = _var(name, np_dtype)
    _add(graph, "Cast", [tensor], casted, name=name, attrs={"to": int(onnx_type)})
    return casted


def _constant_values(tensor: TensorLike) -> Optional[List[int]]:
    gs = _gs()
    if not isinstance(tensor, gs.Constant):
        return None
    return [int(v) for v in np.asarray(tensor.values).reshape(-1)]


def _numel(tensor: TensorLike) -> Optional[int]:
    values = _constant_values(tensor)
    if values is not None:
        return len(values)
    shape = getattr(tensor, "shape", None)
    if not shape:
        return 1 if shape == () or shape == [] else None
    count = 1
    for dim in shape:
        if dim is None or isinstance(dim, str):
            return None
        count *= int(dim)
    return count


def _broadcast_channels(graph: GraphLike, tensor: TensorLike, ref: TensorLike, name: str):
    """Reshape a per-output-channel vector so it broadcasts over ``ref``."""
    count = _numel(tensor)
    if count is None or count <= 1:
        return tensor
    ref_shape = getattr(ref, "shape", None)
    if not ref_shape or int(ref_shape[0] or 0) not in (0, count):
        return None
    target = [count] + [1] * (len(ref_shape) - 1)
    shape = _const(f"{name}_shape", target, np.int64)
    reshaped = _var(f"{name}_bc", np.float32)
    _add(graph, "Reshape", [tensor, shape], reshaped, name=f"{name}_reshape")
    return reshaped


def _dequantize(graph: GraphLike, prefix: str, value: TensorLike, scale: TensorLike, zero_point: Optional[TensorLike], ref: TensorLike) -> Optional[TensorLike]:
    casted = _cast(graph, value, _FLOAT, np.float32, f"{prefix}_f")
    current = casted
    if zero_point is not None:
        zp = _cast(graph, zero_point, _FLOAT, np.float32, f"{prefix}_zp")
        zp = _broadcast_channels(graph, zp, ref, f"{prefix}_zp")
        if zp is None:
            return None
        current = _var(f"{prefix}_centered", np.float32)
        _add(graph, "Sub", [casted, zp], current, name=f"{prefix}_sub")
    scale_b = _broadcast_channels(graph, scale, ref, f"{prefix}_scale")
    if scale_b is None:
        return None
    scaled = _var(f"{prefix}_scaled", np.float32)
    _add(graph, "Mul", [current, scale_b], scaled, name=f"{prefix}_mul")
    return scaled


def _broadcast_ok(param: Optional[TensorLike], ref: TensorLike) -> bool:
    if param is None:
        return True
    return _broadcast_channels_shape_ok(param, ref)


def _broadcast_channels_shape_ok(tensor: TensorLike, ref: TensorLike) -> bool:
    count = _numel(tensor)
    if count is None or count <= 1:
        return True
    ref_shape = getattr(ref, "shape", None)
    return bool(ref_shape) and int(ref_shape[0] or 0) in (0, count)


def _qrange(dtype) -> Optional[tuple]:
    code = _onnx_code(dtype)
    if code == _UINT8:
        return (0.0, 255.0)
    if code == _INT8:
        return (-128.0, 127.0)
    return None


def _requantize(graph: GraphLike, prefix: str, value: TensorLike, scale: TensorLike, zero_point: Optional[TensorLike], output: TensorLike) -> bool:
    bounds = _qrange(output.dtype if output.dtype is not None else None)
    if bounds is None:
        src_code = _onnx_code(output.dtype)
        if src_code is None:
            logger.warning("Cannot requantize %s: output dtype is unknown", prefix)
            return False
    divided = _var(f"{prefix}_div", np.float32)
    _add(graph, "Div", [value, scale], divided, name=f"{prefix}_div")
    rounded = _var(f"{prefix}_round", np.float32)
    _add(graph, "Round", [divided], rounded, name=f"{prefix}_round")
    shifted = rounded
    if zero_point is not None:
        zp = _cast(graph, zero_point, _FLOAT, np.float32, f"{prefix}_yzp")
        shifted = _var(f"{prefix}_shifted", np.float32)
        _add(graph, "Add", [rounded, zp], shifted, name=f"{prefix}_addzp")
    clipped = shifted
    if bounds is not None:
        low = _const(f"{prefix}_qmin", bounds[0], np.float32)
        high = _const(f"{prefix}_qmax", bounds[1], np.float32)
        clipped = _var(f"{prefix}_clip", np.float32)
        _add(graph, "Clip", [shifted, low, high], clipped, name=f"{prefix}_clip")
    code = _onnx_code(output.dtype) or _INT8
    _add(graph, "Cast", [clipped], output, name=f"{prefix}_cast", attrs={"to": int(code)})
    return True


def _conv_attrs(node: NodeLike) -> dict:
    return {key: node.attrs[key] for key in _CONV_ATTRS if key in node.attrs}


def rewrite_bitwise_ops(graph: GraphLike) -> List[str]:
    """Rewrite BOOL bitwise ops, and integer ``BitwiseNot``, into supported ops.

    ``BitwiseNot`` on integers is ``-1 - x`` (two's complement). Integer
    ``BitwiseAnd`` / ``Or`` / ``Xor`` and ``BitShift`` have no TensorRT op.
    """
    rewritten: List[str] = []
    for node in list(graph.nodes):
        name = _node_name(node, f"{node.op}_{id(node)}")
        if node.op in _BOOL_BITWISE and node.inputs and all(_is_bool(t.dtype) for t in node.inputs):
            node.op = _BOOL_BITWISE[node.op]
            node.attrs.clear()
            rewritten.append(name)
            continue
        if node.op != "BitwiseNot" or len(node.outputs) != 1:
            continue
        source = node.inputs[0]
        result = node.outputs[0]
        code = _onnx_code(source.dtype) or _onnx_code(result.dtype)
        if code not in (_INT8, _INT32, _INT64, _UINT8):
            logger.warning("Leaving %s: BitwiseNot dtype is not an integer TensorRT can subtract", name)
            continue
        calc = _INT64 if code == _INT64 else _INT32
        calc_np = np.int64 if calc == _INT64 else np.int32
        _detach(graph, node)
        widened = _cast(graph, source, calc, calc_np, f"{name}_wide")
        neg_one = _const(f"{name}_neg1", -1, calc_np)
        if code == calc:
            _add(graph, "Sub", [neg_one, widened], result, name=f"{name}_sub")
        else:
            temporary = _var(f"{name}_wide_out", calc_np)
            _add(graph, "Sub", [neg_one, widened], temporary, name=f"{name}_sub")
            _add(
                graph,
                "Cast",
                [temporary],
                result,
                name=f"{name}_narrow",
                attrs={"to": int(code)},
            )
        rewritten.append(name)
    return rewritten


def _rewrite_qlinear(graph: GraphLike, node: NodeLike, *, kind: str) -> bool:
    """Fold zero-points into Sub/Mul. TensorRT QuantizeLinear requires zp = 0."""
    name = _node_name(node, kind)
    if kind == "QLinearConv":
        slots = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    else:
        slots = (0, 1, 2, 3, 4, 5, 6, 7)
    tensors = [_optional_input(node, index) for index in slots]
    if kind == "QLinearConv":
        x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, bias = tensors
    else:
        x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp = tensors
        bias = None
    if x is None or x_scale is None or w is None or w_scale is None or y_scale is None:
        logger.warning("Leaving %s: missing QLinear inputs", name)
        return False
    output = node.outputs[0]
    out_dtype = output.dtype if output.dtype is not None else x.dtype
    if _qrange(out_dtype) is None:
        logger.warning("Leaving %s: output dtype is not INT8 or UINT8", name)
        return False
    if not all(_broadcast_ok(param, ref) for param, ref in ((x_scale, x), (x_zp, x), (w_scale, w), (w_zp, w))):
        logger.warning("Leaving %s: per-channel scale rank is unknown", name)
        return False
    _detach(graph, node)
    x_f = _dequantize(graph, f"{name}_x", x, x_scale, x_zp, x)
    w_f = _dequantize(graph, f"{name}_w", w, w_scale, w_zp, w)
    if x_f is None or w_f is None:
        return False
    conv_inputs = [x_f, w_f]
    if bias is not None:
        conv_inputs.append(_cast(graph, bias, _FLOAT, np.float32, f"{name}_bias"))
    conv_out = _var(f"{name}_conv", np.float32)
    op = "Conv" if kind == "QLinearConv" else "MatMul"
    attrs = _conv_attrs(node) if kind == "QLinearConv" else {}
    _add(graph, op, conv_inputs, conv_out, name=f"{name}_{op.lower()}", attrs=attrs)
    output.dtype = out_dtype
    return _requantize(graph, name, conv_out, y_scale, y_zp, output)


def rewrite_qlinear_ops(graph: GraphLike) -> List[str]:
    """Replace ``QLinearConv`` and ``QLinearMatMul`` with float Conv/MatMul."""
    rewritten: List[str] = []
    for node in list(graph.nodes):
        if node.op not in ("QLinearConv", "QLinearMatMul"):
            continue
        name = _node_name(node, node.op)
        if _rewrite_qlinear(graph, node, kind=node.op):
            rewritten.append(name)
    return rewritten


def _rewrite_integer_product(graph: GraphLike, node: NodeLike) -> bool:
    name = _node_name(node, node.op)
    x = _optional_input(node, 0)
    w = _optional_input(node, 1)
    x_zp = _optional_input(node, 2)
    w_zp = _optional_input(node, 3)
    if x is None or w is None or not node.outputs:
        return False
    if not all(_broadcast_ok(param, ref) for param, ref in ((x_zp, x), (w_zp, w))):
        logger.warning("Leaving %s: zero-point rank is unknown", _node_name(node, node.op))
        return False
    output = node.outputs[0]
    attrs = _conv_attrs(node) if node.op == "ConvInteger" else {}
    op = "Conv" if node.op == "ConvInteger" else "MatMul"
    _detach(graph, node)
    one_x = _const(f"{name}_x_scale", 1.0, np.float32)
    one_w = _const(f"{name}_w_scale", 1.0, np.float32)
    x_f = _dequantize(graph, f"{name}_x", x, one_x, x_zp, x)
    w_f = _dequantize(graph, f"{name}_w", w, one_w, w_zp, w)
    if x_f is None or w_f is None:
        return False
    produced = _var(f"{name}_f", np.float32)
    _add(graph, op, [x_f, w_f], produced, name=f"{name}_{op.lower()}", attrs=attrs)
    _add(graph, "Cast", [produced], output, name=f"{name}_i32", attrs={"to": _INT32})
    output.dtype = np.int32
    return True


def rewrite_integer_conv_matmul(graph: GraphLike) -> List[str]:
    """Replace ``ConvInteger`` / ``MatMulInteger`` with a float product cast to INT32.

    The product is exact while the accumulator stays inside the FP32 mantissa
    (24 bits). Large int8 reductions can round.
    """
    rewritten: List[str] = []
    for node in list(graph.nodes):
        if node.op not in ("ConvInteger", "MatMulInteger"):
            continue
        name = _node_name(node, node.op)
        if _rewrite_integer_product(graph, node):
            rewritten.append(name)
    return rewritten


def rewrite_dynamic_quantize_linear(graph: GraphLike) -> List[str]:
    """Expand ``DynamicQuantizeLinear`` into ReduceMin/Max, Round, and Clip.

    TensorRT's parser rejects the fused op and documents this composition.
    The input rank must be known so ReduceMin/Max axes are constant.
    """
    rewritten: List[str] = []
    for node in list(graph.nodes):
        if node.op != "DynamicQuantizeLinear" or len(node.outputs) < 3:
            continue
        name = _node_name(node, "DynamicQuantizeLinear")
        source = node.inputs[0]
        shape = getattr(source, "shape", None)
        if shape is None:
            logger.warning("Leaving %s: input rank is unknown", name)
            continue
        y, y_scale, y_zp = node.outputs[:3]
        rank = len(shape)
        _detach(graph, node)
        if rank == 0:
            x_min = source
            x_max = source
        else:
            axes = list(range(rank))
            x_min = _var(f"{name}_min0", np.float32)
            x_max = _var(f"{name}_max0", np.float32)
            _add(graph, "ReduceMin", [source], x_min, name=f"{name}_rmin", attrs={"axes": axes, "keepdims": 0})
            _add(graph, "ReduceMax", [source], x_max, name=f"{name}_rmax", attrs={"axes": axes, "keepdims": 0})
        zero = _const(f"{name}_zero", 0.0, np.float32)
        x_min_c = _var(f"{name}_min", np.float32)
        x_max_c = _var(f"{name}_max", np.float32)
        _add(graph, "Min", [x_min, zero], x_min_c, name=f"{name}_minc")
        _add(graph, "Max", [x_max, zero], x_max_c, name=f"{name}_maxc")
        span = _var(f"{name}_span", np.float32)
        _add(graph, "Sub", [x_max_c, x_min_c], span, name=f"{name}_span")
        levels = _const(f"{name}_255", 255.0, np.float32)
        raw_scale = _var(f"{name}_raw_scale", np.float32)
        _add(graph, "Div", [span, levels], raw_scale, name=f"{name}_raw_scale")
        one = _const(f"{name}_one", 1.0, np.float32)
        scale_is_zero = _var(f"{name}_scale0", np.bool_)
        _add(graph, "Equal", [raw_scale, zero], scale_is_zero, name=f"{name}_eq")
        _add(graph, "Where", [scale_is_zero, one, raw_scale], y_scale, name=f"{name}_scale")
        y_scale.dtype = np.float32
        neg_min = _var(f"{name}_negmin", np.float32)
        _add(graph, "Sub", [zero, x_min_c], neg_min, name=f"{name}_negmin")
        zp_div = _var(f"{name}_zpdiv", np.float32)
        _add(graph, "Div", [neg_min, y_scale], zp_div, name=f"{name}_zpdiv")
        zp_round = _var(f"{name}_zpround", np.float32)
        _add(graph, "Round", [zp_div], zp_round, name=f"{name}_zpround")
        low = _const(f"{name}_qmin", 0.0, np.float32)
        high = _const(f"{name}_qmax", 255.0, np.float32)
        zp_clip = _var(f"{name}_zpclip", np.float32)
        _add(graph, "Clip", [zp_round, low, high], zp_clip, name=f"{name}_zpclip")
        _add(graph, "Cast", [zp_clip], y_zp, name=f"{name}_zp", attrs={"to": _UINT8})
        y_zp.dtype = np.uint8
        q_div = _var(f"{name}_qdiv", np.float32)
        _add(graph, "Div", [source, y_scale], q_div, name=f"{name}_qdiv")
        q_round = _var(f"{name}_qround", np.float32)
        _add(graph, "Round", [q_div], q_round, name=f"{name}_qround")
        q_shift = _var(f"{name}_qshift", np.float32)
        _add(graph, "Add", [q_round, zp_clip], q_shift, name=f"{name}_qshift")
        q_clip = _var(f"{name}_qclip", np.float32)
        _add(graph, "Clip", [q_shift, low, high], q_clip, name=f"{name}_qclip")
        _add(graph, "Cast", [q_clip], y, name=f"{name}_y", attrs={"to": _UINT8})
        y.dtype = np.uint8
        rewritten.append(name)
    return rewritten


def rewrite_center_crop_pad(graph: GraphLike) -> List[str]:
    """Replace static ``CenterCropPad`` with Slice and Pad."""
    rewritten: List[str] = []
    for node in list(graph.nodes):
        if node.op != "CenterCropPad" or len(node.inputs) < 2 or not node.outputs:
            continue
        name = _node_name(node, "CenterCropPad")
        source = node.inputs[0]
        target = _constant_values(node.inputs[1])
        in_shape = getattr(source, "shape", None)
        if target is None or not in_shape or any(dim is None or isinstance(dim, str) for dim in in_shape):
            logger.warning("Leaving %s: input shape or target shape is not constant", name)
            continue
        rank = len(in_shape)
        axes = node.attrs.get("axes")
        if axes is None:
            axes = list(range(rank - len(target), rank))
        axes = [int(axis) for axis in axes]
        if len(axes) != len(target):
            logger.warning("Leaving %s: axes and target shape differ in length", name)
            continue
        output = node.outputs[0]
        starts = [0] * rank
        ends = [int(dim) for dim in in_shape]
        pad_begin = [0] * rank
        pad_end = [0] * rank
        for axis, size in zip(axes, target):
            if axis < 0:
                axis += rank
            diff = int(in_shape[axis]) - int(size)
            if diff > 0:
                starts[axis] = diff // 2
                ends[axis] = starts[axis] + int(size)
            elif diff < 0:
                pad_begin[axis] = (-diff) // 2
                pad_end[axis] = (-diff) - pad_begin[axis]
        needs_slice = starts != [0] * rank or ends != [int(dim) for dim in in_shape]
        needs_pad = any(pad_begin) or any(pad_end)
        _detach(graph, node)
        current = source
        if needs_slice:
            sliced = output if not needs_pad else _var(f"{name}_sliced", source.dtype)
            _add(
                graph,
                "Slice",
                [
                    current,
                    _const(f"{name}_starts", starts, np.int64),
                    _const(f"{name}_ends", ends, np.int64),
                    _const(f"{name}_axes", list(range(rank)), np.int64),
                    _const(f"{name}_steps", [1] * rank, np.int64),
                ],
                sliced,
                name=f"{name}_slice",
            )
            current = sliced
        if needs_pad:
            pad_dtype = _np_dtype(source.dtype) or np.dtype(np.float32)
            _add(
                graph,
                "Pad",
                [
                    current,
                    _const(f"{name}_pads", pad_begin + pad_end, np.int64),
                    _const(f"{name}_padvalue", 0, pad_dtype),
                ],
                output,
                name=f"{name}_pad",
                attrs={"mode": "constant"},
            )
        elif not needs_slice:
            _add(graph, "Identity", [current], output, name=f"{name}_id")
        rewritten.append(name)
    return rewritten


def rewrite_unsupported_onnx_ops(graph: GraphLike) -> List[str]:
    """Rewrite TensorRT-unsupported ops that have a supported decomposition."""
    rewritten: List[str] = []
    rewritten.extend(rewrite_bitwise_ops(graph))
    rewritten.extend(rewrite_qlinear_ops(graph))
    rewritten.extend(rewrite_integer_conv_matmul(graph))
    rewritten.extend(rewrite_dynamic_quantize_linear(graph))
    rewritten.extend(rewrite_center_crop_pad(graph))
    if rewritten:
        logger.info("Rewrote TensorRT-unsupported ops: %s", rewritten)
    return rewritten
