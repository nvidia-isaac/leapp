#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""ONNX dynamo lowering for ``leapp::warp_runner``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from leapp.utils.logging import _get_logger

from .schema import (
    QUALIFIED_NAME,
    ONNX_WRP_DOMAIN,
    ONNX_WRP_OPSET,
    ONNX_WRP_OP_TYPE,
    decode_output_dtypes,
    decode_output_shapes,
    get_op,
    _resolve_dtype,
)

_GLOBAL_ONNX_TRANSLATIONS: dict[Any, Callable[..., Any]] = {}
_ONNX_EXPORT_PATCHED = False


def lower_warp_runner_to_onnx(
    inputs,
    output_shapes,
    output_dtypes,
    output_mask,
    bundle,
):
    """Lower ``leapp.warp_runner`` to ``com.nvidia.warp::WrpRunner`` during ONNX export."""
    from onnxscript import ir
    from torch.onnx._internal.exporter import _core, _tensors

    tracer = _core.current_tracer
    if tracer is None:
        raise RuntimeError(
            f"Cannot lower {QUALIFIED_NAME}: ONNX export tracer is not active."
        )

    shape_lists = decode_output_shapes(output_shapes)
    dtype_lists = decode_output_dtypes(output_dtypes)
    if len(shape_lists) != len(dtype_lists):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_shapes ({len(shape_lists)}) and "
            f"output_dtypes ({len(dtype_lists)}) must have equal length"
        )

    data_inputs = list(inputs)
    dtypes = [_resolve_dtype(name) for name in dtype_lists]
    shapes = [tuple(int(dim) for dim in shape) for shape in shape_lists]

    attrs = {
        "input_names": ",".join(f"input_{i}" for i in range(len(data_inputs))),
        "output_names": ",".join(f"output_{i}" for i in range(len(shapes))),
        "output_shape": output_shapes,
    }

    wrp_inputs = [*data_inputs, bundle]

    outputs = [_tensors.SymbolicTensor(tracer.opset) for _ in range(len(shapes))]
    for output, shape, dtype in zip(outputs, shapes, dtypes):
        output.dtype = _core._TORCH_DTYPE_TO_ONNX[dtype]
        output.shape = ir.Shape(shape)

    node = ir.Node(
        ONNX_WRP_DOMAIN,
        ONNX_WRP_OP_TYPE,
        inputs=wrp_inputs,
        attributes=ir.convenience.convert_attributes(attrs),
        outputs=outputs,
        version=ONNX_WRP_OPSET,
    )
    tracer.nodes.append(node)

    if len(outputs) == 1:
        return outputs[0]
    return outputs


def _patch_torch_onnx_export() -> None:
    """Merge LEAPP custom ONNX lowerings into every ``torch.onnx.export`` call."""
    global _ONNX_EXPORT_PATCHED
    if _ONNX_EXPORT_PATCHED:
        return

    original_export = torch.onnx.export

    @functools.wraps(original_export)
    def export_with_leapp_custom_ops(*args, custom_translation_table=None, **kwargs):
        merged = dict(_GLOBAL_ONNX_TRANSLATIONS)
        if custom_translation_table:
            merged.update(custom_translation_table)
        return original_export(
            *args,
            custom_translation_table=merged or None,
            **kwargs,
        )

    torch.onnx.export = export_with_leapp_custom_ops
    _ONNX_EXPORT_PATCHED = True


def register_onnx_lowering() -> None:
    """Register the Warp segment ONNX lowering globally for dynamo export."""
    _GLOBAL_ONNX_TRANSLATIONS[get_op().default] = lower_warp_runner_to_onnx
    _patch_torch_onnx_export()
    _get_logger().debug(
        f"Registered global ONNX lowering for {QUALIFIED_NAME} -> "
        f"{ONNX_WRP_DOMAIN}::{ONNX_WRP_OP_TYPE}"
    )
