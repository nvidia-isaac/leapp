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

from .metadata import decode_runtime_metadata, output_dtypes, output_shapes
from .schema import (
    QUALIFIED_NAME,
    ONNX_WRP_DOMAIN,
    ONNX_WRP_OPSET,
    ONNX_WRP_OP_TYPE,
    encode_output_mask,
    get_op,
    _format_output_shape_attr,
    _resolve_dtype,
)

_GLOBAL_ONNX_TRANSLATIONS: dict[Any, Callable[..., Any]] = {}
_ONNX_EXPORT_PATCHED = False


def _masked_output_names(metadata: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for spec in metadata.get("outputs", []):
        if spec.get("mask", True):
            name = spec.get("param_name")
            if name:
                names.append(str(name))
    return names


def lower_warp_runner_to_onnx(
    inputs,
    runtime_metadata,
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

    metadata = decode_runtime_metadata(runtime_metadata)
    shape_lists = output_shapes(metadata)
    dtype_lists = output_dtypes(metadata)
    if len(shape_lists) != len(dtype_lists):
        raise ValueError(
            f"{QUALIFIED_NAME}: runtime_metadata output shapes ({len(shape_lists)}) "
            f"and dtypes ({len(dtype_lists)}) must have equal length"
        )

    data_inputs = list(inputs)
    dtypes = [_resolve_dtype(name) for name in dtype_lists]
    shapes = [tuple(int(dim) for dim in shape) for shape in shape_lists]
    output_names = _masked_output_names(metadata)

    attrs = {
        "runtime_metadata": runtime_metadata,
        # Legacy attrs are kept while the C++ runtime migrates to runtime_metadata.
        "wrp_name": str(metadata.get("wrp_name", "")),
        "input_names": ",".join(
            str(spec.get("param_name", f"input_{i}"))
            for i, spec in enumerate(metadata.get("inputs", []))
        ),
        "output_names": ",".join(output_names),
        "output_shape": _format_output_shape_attr(shape_lists),
        "output_mask": encode_output_mask(
            [bool(spec.get("mask", True)) for spec in metadata.get("outputs", [])]
        ),
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
