#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Registration of the ``leapp::warp_runner`` custom operator.

Implementation is split by backend:
- ``schema.py`` — op identity and encode/decode contract
- ``fx.py`` — fake/eager PyTorch kernels for tracing and ``torch.export``
- ``onnx.py`` — dynamo ONNX lowering to ``com.nvidia.warp::WrpRunner``
- ``bundle.py`` — WRPB archive packing (embedded at trace time)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from leapp.utils.logging import _get_logger

from ..registry import register_export_hooks
from . import fx as fx_impl
from . import onnx as onnx_impl
from .metadata import (
    build_runtime_metadata,
    decode_runtime_metadata,
    encode_runtime_metadata,
    output_dtypes as runtime_output_dtypes,
    output_mask as runtime_output_mask,
    output_shapes as runtime_output_shapes,
)
from .schema import (
    NAMESPACE,
    OP_NAME,
    QUALIFIED_NAME,
    ONNX_WRP_DOMAIN,
    ONNX_WRP_OPSET,
    ONNX_WRP_OP_TYPE,
    _SCHEMA,
    get_op,
)

if TYPE_CHECKING:
    import onnx
    import torch.nn

from leapp.leapp_graph.datatypes.warp import wp

_LIB = torch.library.Library(NAMESPACE, "FRAGMENT")

_SUPPORTED_EXPORT_BACKENDS = frozenset({"onnx-dynamo", "exported-program"})


def module_contains_warp_runner(module: "torch.nn.Module | None") -> bool:
    if module is None:
        return False
    op_packet = get_op()
    for _, submodule in module.named_modules():
        graph = getattr(submodule, "graph", None)
        if graph is None:
            continue
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            if target is op_packet:
                return True
            if (
                isinstance(target, torch._ops.OpOverload)
                and target.overloadpacket is op_packet
            ):
                return True
    return False


def _onnx_graph_contains_warp_runner(graph) -> bool:
    for node in graph.node:
        if node.domain == ONNX_WRP_DOMAIN and node.op_type == ONNX_WRP_OP_TYPE:
            return True
        for attribute in node.attribute:
            if attribute.HasField("g") and _onnx_graph_contains_warp_runner(attribute.g):
                return True
            if any(_onnx_graph_contains_warp_runner(graph) for graph in attribute.graphs):
                return True
    return False


def onnx_model_contains_warp_runner(model: "onnx.ModelProto | None") -> bool:
    return model is not None and _onnx_graph_contains_warp_runner(model.graph)


def _warp_unsupported_message(backend: str) -> str:
    return (
        f"export_with='{backend}' is not supported for graphs containing a Warp "
        f"segment ({QUALIFIED_NAME}). Use export_with='onnx-dynamo' or "
        "export_with='exported-program' (alias 'pt2') to embed the APIC bundle "
        "as a constant input."
    )


def _register_export_hooks() -> None:
    register_export_hooks(
        op_name=QUALIFIED_NAME,
        detect_in_module=module_contains_warp_runner,
        supported_backends=_SUPPORTED_EXPORT_BACKENDS,
        unsupported_message=_warp_unsupported_message,
    )


def _register() -> None:
    """Idempotently register the PyTorch op, eager kernel, and ONNX lowerings."""
    if not (
        hasattr(torch.ops, NAMESPACE)
        and hasattr(getattr(torch.ops, NAMESPACE), OP_NAME)
    ):
        _LIB.define(_SCHEMA)
        torch.library.register_fake(
            QUALIFIED_NAME, fx_impl.warp_runner_fake, lib=_LIB
        )
        _LIB.impl(OP_NAME, fx_impl.warp_runner_eager, "CompositeExplicitAutograd")

        _get_logger().debug(
            f"Registered custom op {QUALIFIED_NAME} with schema: {_SCHEMA}"
        )

    if wp is not None:
        onnx_impl.register_onnx_lowering()
    _register_export_hooks()


_register()

__all__ = [
    "NAMESPACE",
    "OP_NAME",
    "QUALIFIED_NAME",
    "ONNX_WRP_DOMAIN",
    "ONNX_WRP_OPSET",
    "ONNX_WRP_OP_TYPE",
    "module_contains_warp_runner",
    "onnx_model_contains_warp_runner",
    "get_op",
    "build_runtime_metadata",
    "encode_runtime_metadata",
    "decode_runtime_metadata",
    "runtime_output_shapes",
    "runtime_output_dtypes",
    "runtime_output_mask",
]
