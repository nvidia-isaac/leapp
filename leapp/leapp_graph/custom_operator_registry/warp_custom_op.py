#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Registration of the ``leapp::warp_runner`` custom operator.

This module is the single home for the Warp segment placeholder op. LEAPP's
Warp tracer (``traced_node.py``) emits ``leapp.warp_runner`` nodes into the FX
graph; this file defines that op and lowers it to ONNX at export time.

TODO — gaps before a Warp segment is fully exportable end-to-end
---------------------------------------------------------------
The current implementation is enough to trace, export an ONNX graph containing
``com.nvidia.warp::WrpRunner``, and inspect the result. The following are still
missing if the goal is a runnable exported pipeline:

- **``.wrp`` payload** — deferred until after FX pruning in ``_save_warp_captures``;
  embedded into ONNX via ``_embed_warp_bundles`` on the dynamo export path.
- **Real I/O names** — ONNX ``input_names`` / ``output_names`` are filled from
  the segment save plan at embed time (live inputs/outputs only).
- **``output_mask`` in lowering** — unused outputs stay in the ONNX node with
  zero shape; APIC ``capture_save`` only registers live outputs.
- **Embedded bundle mode** — canonical: ``wrp_name`` plus a uint8 tensor input
  carrying the WRPB archive (self-contained ONNX artifacts).
- **Portable paths** — ``path`` is a logical storage namespace
  (``leapp/<graph>/<node>/<proxy>``), not a filesystem path.
- **ORT custom op** — vanilla ONNX Runtime does not register ``WrpRunner``; the
  prototype C++/CUDA kernel must be installed before inference or LEAPP
  validation can succeed.
- **Variable segment arity** — the ORT prototype may require fixed I/O counts per
  op registration; multi-segment graphs with different arities may need multiple
  WrpRunner variants or a more general runtime op.

The schema is intentionally restricted to types that parse on the minimum Torch
version LEAPP supports (``torch>=2.6.0``).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from leapp.utils.logging import _get_logger

from .registry import register_export_hooks

try:
    import warp as wp
except ImportError:
    wp = None


# =============================================================================
# SHARED — operator identity, schema, and helpers used by both FX and ONNX
# =============================================================================
# These symbols are the contract between this module, ``traced_node.py`` (which
# emits FX nodes), and the ONNX lowering below.

NAMESPACE = "leapp"
OP_NAME = "warp_runner"
QUALIFIED_NAME = f"{NAMESPACE}::{OP_NAME}"

# Op schema — one finalized Warp/APIC segment.
#
#   inputs         : traced input tensors for the segment (variadic Tensor[]).
#   output_shapes  : per-output shape; one inner list per segment output.
#   output_dtypes  : per-output dtype name (e.g. "float32"), parallel to
#                    ``output_shapes``. String names are used instead of
#                    ``ScalarType`` because PyTorch surfaces schema ScalarType
#                    args to Python as opaque integer codes with no stable
#                    public inverse.
#   path           : runtime storage namespace for the segment bundle, e.g.
#                    ``leapp/<graph_name>/<node_name>/<proxy_name>``; set when
#                    the segment marker is emitted in ``close_warp_segment``.
#   output_mask    : per-output flag for which outputs are live graph outputs;
#                    rewritten by the post-prune pass in ``traced_node.py``.
_SCHEMA = (
    f"{OP_NAME}(Tensor[] inputs, int[][] output_shapes, "
    f"str[] output_dtypes, str path, bool[] output_mask) -> Tensor[]"
)

# ONNX export target — referenced by the eager guard impl and the lowering below.
ONNX_WRP_DOMAIN = "com.nvidia.warp"
ONNX_WRP_OPSET = 1
ONNX_WRP_OP_TYPE = "WrpRunner"

# Keep a module-level handle so the Library (and thus the registration) is not
# garbage collected for the lifetime of the process.
_LIB = torch.library.Library(NAMESPACE, "FRAGMENT")


def get_op() -> "torch._ops.OpOverloadPacket":
    """Return the registered ``warp_runner`` overload packet.

    Called by ``traced_node.py`` when closing a Warp segment to emit the FX
    node: ``create_proxy("call_function", get_op().default, (...))``.
    """
    return getattr(getattr(torch.ops, NAMESPACE), OP_NAME)


def _resolve_dtype(name: str) -> torch.dtype:
    """Resolve a dtype name (e.g. "float32") to a ``torch.dtype``."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{QUALIFIED_NAME}: unknown output dtype name '{name}'")
    return dtype


# =============================================================================
# FX GRAPH / TRACING — PyTorch custom op implementations
# =============================================================================
# These kernels are what make ``leapp.warp_runner`` a valid FX node during LEAPP
# tracing and ``torch.export``. They do not execute real Warp work; execution is
# delegated to the ORT ``WrpRunner`` kernel after ONNX export.
#
# FX emission lives in ``traced_node.py`` (``_close_warp_segment``): it builds
# ``(inputs, output_shapes, output_dtypes, path, output_mask)`` and attaches
# ``node.meta["leapp_warp_segment"]`` for downstream export passes.


def _warp_runner_fake(inputs, output_shapes, output_dtypes, path, output_mask):
    """Abstract impl: produce correctly-shaped meta outputs from the spec.

    A ``Tensor[] -> Tensor[]`` op cannot infer its output count/shapes from the
    inputs alone, so the output specification is passed explicitly and used here
    to build the fake outputs. The number, shapes and dtypes returned mirror
    ``output_shapes`` / ``output_dtypes`` exactly.
    """
    if len(output_shapes) != len(output_dtypes):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_shapes ({len(output_shapes)}) and "
            f"output_dtypes ({len(output_dtypes)}) must have equal length"
        )
    if output_mask and len(output_mask) != len(output_shapes):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_mask ({len(output_mask)}) must match the "
            f"number of outputs ({len(output_shapes)}) when provided"
        )

    device = inputs[0].device if len(inputs) > 0 else torch.device("cpu")
    return [
        torch.empty(list(shape), dtype=_resolve_dtype(name), device=device)
        for shape, name in zip(output_shapes, output_dtypes)
    ]


def _warp_runner_eager(inputs, output_shapes, output_dtypes, path, output_mask):
    """Eager kernel: allocate shape-correct (zeros) outputs from the spec.

    This does **not** perform the Warp computation — real execution is the ORT/C++
    ``WrpRunner`` kernel after ONNX export. It exists so the op is callable in eager
    mode (keeping the op valid during ``torch.export`` tracing). The placeholder
    zeros are why eager validation reports a value mismatch until the real runtime
    kernel exists.
    """
    if len(output_shapes) != len(output_dtypes):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_shapes ({len(output_shapes)}) and "
            f"output_dtypes ({len(output_dtypes)}) must have equal length"
        )
    device = inputs[0].device if len(inputs) > 0 else torch.device("cpu")
    return [
        torch.zeros(list(shape), dtype=_resolve_dtype(name), device=device)
        for shape, name in zip(output_shapes, output_dtypes)
    ]


# =============================================================================
# ONNX EXPORT — dynamo lowering and global registration
# =============================================================================
# When warp-lang is available, importing this module patches ``torch.onnx.export``
# so every dynamo export automatically includes the translation table entry for
# ``leapp.warp_runner`` -> ``com.nvidia.warp::WrpRunner``. No backend-specific
# wiring in ``onnx_export_backend.py`` is required.

_GLOBAL_ONNX_TRANSLATIONS: dict[Any, Callable[..., Any]] = {}
_ONNX_EXPORT_PATCHED = False


def _format_output_shape_attr(output_shapes: list[list[int]]) -> str:
    """Format per-output shapes for the WrpRunner ``output_shape`` attribute."""
    parts = []
    for shape in output_shapes:
        if not shape:
            parts.append("0")
            continue
        parts.append(",".join(str(int(dim)) for dim in shape))
    return ";".join(parts)


def lower_warp_runner_to_onnx(
    inputs,
    output_shapes,
    output_dtypes,
    path,
    output_mask,
):
    """Lower ``leapp.warp_runner`` to ``com.nvidia.warp::WrpRunner`` during ONNX export.

    Invoked by the dynamo ONNX exporter via the global translation table. Builds
    an ``onnxscript.ir.Node`` directly because the return type is variadic
    (``Tensor[]``).
    """
    from onnxscript import ir
    from torch.onnx._internal.exporter import _core, _tensors

    tracer = _core.current_tracer
    if tracer is None:
        raise RuntimeError(
            f"Cannot lower {QUALIFIED_NAME}: ONNX export tracer is not active."
        )

    if len(output_shapes) != len(output_dtypes):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_shapes ({len(output_shapes)}) and "
            f"output_dtypes ({len(output_dtypes)}) must have equal length"
        )

    data_inputs = list(inputs)
    dtypes = [_resolve_dtype(name) for name in output_dtypes]
    shapes = [tuple(int(dim) for dim in shape) for shape in output_shapes]

    # Names and shapes are finalized at embed time from the segment save plan.
    attrs = {
        "storage_path": path or "",
        "input_names": ",".join(f"input_{i}" for i in range(len(data_inputs))),
        "output_names": ",".join(f"output_{i}" for i in range(len(shapes))),
        "output_shape": _format_output_shape_attr(
            [list(shape) for shape in output_shapes]
        ),
    }

    outputs = [_tensors.SymbolicTensor(tracer.opset) for _ in range(len(shapes))]
    for output, shape, dtype in zip(outputs, shapes, dtypes):
        output.dtype = _core._TORCH_DTYPE_TO_ONNX[dtype]
        output.shape = ir.Shape(shape)

    node = ir.Node(
        ONNX_WRP_DOMAIN,
        ONNX_WRP_OP_TYPE,
        inputs=data_inputs,
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


def _register_onnx_lowering() -> None:
    """Register the Warp segment ONNX lowering globally for dynamo export."""
    if wp is None:
        return

    _GLOBAL_ONNX_TRANSLATIONS[get_op().default] = lower_warp_runner_to_onnx
    _patch_torch_onnx_export()
    _get_logger().debug(
        f"Registered global ONNX lowering for {QUALIFIED_NAME} -> "
        f"{ONNX_WRP_DOMAIN}::{ONNX_WRP_OP_TYPE}"
    )


# =============================================================================
# EXPORT BACKEND SUPPORT — which export paths can carry a Warp segment
# =============================================================================
# Supported:
#   * onnx-dynamo  — ONNX via ``torch.onnx.export(dynamo=True)``
#
# NOT supported (the export registry rejects these backends when a segment is
# present): ``jit-script``, ``jit-trace``, ``onnx-torchscript``.
#
# Why only the dynamo ONNX path
# -----------------------------
# ``warp_runner`` is a *variadic, index-addressed* multi-output node:
# ``(Tensor[] inputs, int[][] output_shapes, ...) -> Tensor[]`` whose downstream
# consumers read ``operator.getitem(node, i)``. The dynamo exporter keeps it as a
# real multi-output node and maps each ``getitem`` to the i-th output directly,
# mixed dtypes included (see ``lower_warp_runner_to_onnx``), and the embedded
# WRPB bundle is attached as a ``uint8`` initializer at export time.
#
# The legacy TorchScript tracer/ONNX exporter cannot represent this op
# (``int[][]`` args break tracing, list-typed constants break the exporter, and
# the ``Tensor[]`` return only fits a homogeneous ONNX sequence). ``jit-script``
# could carry the op structurally but produces an *incomplete* artifact: it has
# no mechanism to embed the APIC bundle, so the ``.pt`` would script and run the
# placeholder eager kernel (zeros) with no Warp data. It is therefore rejected
# rather than silently emitting a broken model. A future ``torch.export``/``.pt2``
# backend is the intended non-ONNX path (lifted-constant bundle + unified C++ op).

_SUPPORTED_EXPORT_BACKENDS = frozenset({"onnx-dynamo"})


def _module_contains_warp_runner(module: "torch.nn.Module") -> bool:
    """True if any GraphModule in ``module`` calls ``leapp::warp_runner``."""
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


def _warp_unsupported_message(backend: str) -> str:
    return (
        f"export_with='{backend}' is not supported for graphs containing a Warp "
        f"segment ({QUALIFIED_NAME}). Only export_with='onnx-dynamo' can embed the "
        "APIC bundle and represent the op's variadic, mixed-dtype, index-addressed "
        "outputs. Use export_with='onnx-dynamo'."
    )


def _register_export_hooks() -> None:
    register_export_hooks(
        op_name=QUALIFIED_NAME,
        detect_in_module=_module_contains_warp_runner,
        supported_backends=_SUPPORTED_EXPORT_BACKENDS,
        unsupported_message=_warp_unsupported_message,
    )


# =============================================================================
# BOOTSTRAP — import-time registration
# =============================================================================


def _register() -> None:
    """Idempotently register the PyTorch op, eager kernel, and ONNX lowerings."""
    if not (
        hasattr(torch.ops, NAMESPACE)
        and hasattr(getattr(torch.ops, NAMESPACE), OP_NAME)
    ):
        _LIB.define(_SCHEMA)
        torch.library.register_fake(QUALIFIED_NAME, _warp_runner_fake, lib=_LIB)
        # Shape-correct eager kernel so the op stays callable during
        # ``torch.export`` tracing; real execution is the ORT WrpRunner kernel
        # after export.
        _LIB.impl(OP_NAME, _warp_runner_eager, "CompositeExplicitAutograd")

        _get_logger().debug(
            f"Registered custom op {QUALIFIED_NAME} with schema: {_SCHEMA}"
        )

    _register_onnx_lowering()
    _register_export_hooks()


_register()
