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

- **``.wrp`` payload** — segments are not serialized yet. The op's ``path``
  argument and the ONNX ``wrp_path`` attribute are empty placeholders; export
  should call ``wp.capture_save`` (or an embedded-bundle path) and fill them in.
- **Real I/O names** — ONNX ``input_names`` / ``output_names`` are synthetic
  (``input_0``, ``output_0``, …). They should come from the segment's Warp
  kernel parameter names (``segment.input_refs`` / ``output_refs``).
- **``output_mask`` in lowering** — the mask is stamped on FX nodes during
  pruning but not yet applied when building WrpRunner attributes (unused outputs
  should be dropped or zero-shaped consistently with the runtime contract).
- **Embedded bundle mode** — alternative to path-based export: ``wrp_name`` plus
  a uint8 tensor input carrying the serialized ``.wrp`` bytes (for self-contained
  ONNX artifacts).
- **Portable paths** — ``wrp_path`` is currently absolute; may need to be
  relative to the ONNX file or replaced entirely by embedded mode.
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
#   path           : path to the segment's ``.wrp`` payload; empty until export
#                    materializes it (see module TODO).
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
    """Eager kernel: allocate shape-correct outputs from the spec.

    This does **not** perform the Warp computation — real execution is the ORT/C++
    ``WrpRunner`` kernel after ONNX export. It exists so the op survives
    ``torch.jit.trace`` (which executes the forward on real inputs): trace records
    a single ``leapp::warp_runner`` node and the TorchScript ONNX symbolic lowers
    it to ``com.nvidia.warp::WrpRunner``. Values are zeros placeholders.
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

    # TODO: read real names from segment metadata; honor output_mask; fill wrp_path
    # from a serialized .wrp bundle (see module docstring TODO list).
    attrs = {
        "wrp_path": path or "",
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
# ONNX EXPORT (TorchScript) — legacy symbolic registration
# =============================================================================
# The TorchScript ONNX exporter (``torch.onnx.export(..., dynamo=False)``) uses a
# different extension mechanism than dynamo: a "symbolic" registered globally via
# ``register_custom_op_symbolic``. It builds the same ``com.nvidia.warp::WrpRunner``
# node against the TorchScript graph builder ``g`` instead of ``onnxscript.ir``.
#
# This path requires the op to reach export without raising: under ``jit-trace``
# the eager kernel above produces shape-correct outputs; under ``jit-script`` the
# op is recorded without execution. The custom-domain opset import defaults to 1.

_TORCHSCRIPT_SYMBOLIC_REGISTERED = False


def _ts_parse_int_list_list(value) -> list[list[int]]:
    """Parse an ``int[][]`` schema arg (a nested ``prim::ListConstruct``)."""
    from torch.onnx import symbolic_helper

    return [
        symbolic_helper._parse_arg(inner, "is") for inner in value.node().inputs()
    ]


def _ts_parse_str_list(value) -> list[str]:
    """Parse a ``str[]`` schema arg (a ``prim::ListConstruct`` of strings)."""
    from torch.onnx import symbolic_helper

    return [
        symbolic_helper._parse_arg(inner, "s") for inner in value.node().inputs()
    ]


def _ts_set_output_type(value, torch_dtype: torch.dtype, shape) -> None:
    """Best-effort: stamp dtype/shape on a symbolic output so the ONNX graph is typed."""
    try:
        value.setType(value.type().with_dtype(torch_dtype).with_sizes(list(shape)))
    except Exception:
        try:
            value.setType(
                torch._C.TensorType.create_from_tensor(
                    torch.zeros(list(shape), dtype=torch_dtype)
                )
            )
        except Exception:
            pass


def warp_runner_symbolic(g, inputs, output_shapes, output_dtypes, path, output_mask):
    """TorchScript ONNX symbolic: lower ``leapp::warp_runner`` to ``WrpRunner``.

    Mirrors :func:`lower_warp_runner_to_onnx` (the dynamo lowering) but builds the
    node via the TorchScript graph builder ``g``. Variadic ``Tensor[]`` inputs are
    unpacked from their ``prim::ListConstruct``; output count comes from the static
    ``output_shapes`` so ``g.op(..., outputs=N)`` returns the right arity.
    """
    from torch.onnx import symbolic_helper

    data_inputs = symbolic_helper._unpack_list(inputs)
    shapes = _ts_parse_int_list_list(output_shapes)
    dtype_names = _ts_parse_str_list(output_dtypes)
    wrp_path = symbolic_helper._parse_arg(path, "s")
    num_outputs = len(shapes)

    # TODO: read real names from segment metadata; honor output_mask; fill wrp_path
    # from a serialized .wrp bundle (see module docstring TODO list).
    attrs = {
        "wrp_path_s": wrp_path or "",
        "input_names_s": ",".join(f"input_{i}" for i in range(len(data_inputs))),
        "output_names_s": ",".join(f"output_{i}" for i in range(num_outputs)),
        "output_shape_s": _format_output_shape_attr(shapes),
    }

    outputs = g.op(
        f"{ONNX_WRP_DOMAIN}::{ONNX_WRP_OP_TYPE}",
        *data_inputs,
        outputs=num_outputs,
        **attrs,
    )
    output_values = list(outputs) if num_outputs > 1 else [outputs]

    torch_dtypes = [_resolve_dtype(name) for name in dtype_names]
    for value, shape, dtype in zip(output_values, shapes, torch_dtypes):
        _ts_set_output_type(value, dtype, shape)

    return output_values[0] if num_outputs == 1 else output_values


def _register_torchscript_symbolic() -> None:
    """Register the TorchScript ONNX symbolic across the supported opset range."""
    global _TORCHSCRIPT_SYMBOLIC_REGISTERED
    if _TORCHSCRIPT_SYMBOLIC_REGISTERED:
        return
    # Custom-domain ops are opset-agnostic; register broadly so any export opset
    # the backend chooses resolves the symbolic.
    for opset in range(9, 24):
        torch.onnx.register_custom_op_symbolic(
            QUALIFIED_NAME, warp_runner_symbolic, opset
        )
    _TORCHSCRIPT_SYMBOLIC_REGISTERED = True
    _get_logger().debug(
        f"Registered TorchScript ONNX symbolic for {QUALIFIED_NAME} -> "
        f"{ONNX_WRP_DOMAIN}::{ONNX_WRP_OP_TYPE}"
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
        # Shape-correct eager kernel so the op survives jit-trace; real execution
        # is the ORT WrpRunner kernel after export.
        _LIB.impl(OP_NAME, _warp_runner_eager, "CompositeExplicitAutograd")

        _get_logger().debug(
            f"Registered custom op {QUALIFIED_NAME} with schema: {_SCHEMA}"
        )

    _register_onnx_lowering()
    _register_torchscript_symbolic()


_register()
