#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Registration of the ``leapp::warp_runner`` custom operator.

This module registers a single PyTorch custom op that stands in for one
finalized Warp/APIC segment in the traced FX graph. Emitting this op
lets the segment be a first-class node that downstream export can lower
directly to the ONNX ``com.nvidia.warp::WrpRunner`` custom op, with no
separate marker-replacement pass.

Only the *registration* lives here:
- the op schema,
- a fake/abstract implementation so the op traces cleanly under FakeTensorMode,
  ``torch.export``, and the dynamo ONNX exporter,
- a guard eager implementation that raises, because the real execution is the
  ORT/C++ ``WrpRunner`` kernel, not a Torch kernel.

The schema is intentionally restricted to types that parse on the minimum Torch
version LEAPP supports (``torch>=2.6.0``); see the matrix verified in review.
"""

import torch

from leapp.utils.logging import _get_logger

# Operator identity.
NAMESPACE = "leapp"
OP_NAME = "warp_runner"
QUALIFIED_NAME = f"{NAMESPACE}::{OP_NAME}"

# Schema for one Warp segment.
#
#   inputs         : the segment's traced input tensors (variadic, N-ary).
#   output_shapes  : per-output shape, one inner list per produced output.
#   output_dtypes  : per-output dtype name (e.g. "float32"), parallel to
#                    ``output_shapes``. Passed as a string rather than
#                    ``ScalarType`` because PyTorch surfaces ScalarType schema
#                    args to Python as opaque integer enum codes with no stable
#                    public inverse; the name round-trips cleanly via
#                    ``getattr(torch, name)``.
#   path           : filesystem path associated with the segment's payload
#                    (e.g. the saved ``.wrp`` bundle); empty string when the
#                    payload is embedded rather than referenced by path.
#   output_mask    : per-output boolean flag carried for the runtime/lowering
#                    (e.g. which candidate outputs are live graph outputs).
#
# Static ONNX-only attributes (wrp_name, input/output param names, etc.) are not
# part of the schema; they ride in ``node.meta`` and are materialized by the
# export-time lowering.
_SCHEMA = (
    f"{OP_NAME}(Tensor[] inputs, int[][] output_shapes, "
    f"str[] output_dtypes, str path, bool[] output_mask) -> Tensor[]"
)

# Keep a module-level handle so the Library (and thus the registration) is not
# garbage collected for the lifetime of the process.
_LIB = torch.library.Library(NAMESPACE, "FRAGMENT")


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


def _resolve_dtype(name: str) -> torch.dtype:
    """Resolve a dtype name (e.g. "float32") to a ``torch.dtype``."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{QUALIFIED_NAME}: unknown output dtype name '{name}'")
    return dtype


def _warp_runner_not_implemented(inputs, output_shapes, output_dtypes, path, output_mask):
    """Guard eager impl. The real execution is the ORT/C++ WrpRunner kernel."""
    raise NotImplementedError(
        f"{QUALIFIED_NAME} has no eager Torch implementation; it is lowered to "
        f"the ONNX 'com.nvidia.warp::WrpRunner' custom op at export time."
    )


def _register() -> None:
    """Idempotently register the op, its fake impl, and the guard impl."""
    if hasattr(torch.ops, NAMESPACE) and hasattr(getattr(torch.ops, NAMESPACE), OP_NAME):
        # Already registered (e.g. module re-import); nothing to do.
        return

    _LIB.define(_SCHEMA)
    torch.library.register_fake(QUALIFIED_NAME, _warp_runner_fake, lib=_LIB)
    # No device-specific kernel exists; route every backend to the guard so an
    # accidental eager call fails with a clear message instead of a missing
    # kernel error.
    _LIB.impl(OP_NAME, _warp_runner_not_implemented, "CompositeExplicitAutograd")

    _get_logger().debug(f"Registered custom op {QUALIFIED_NAME} with schema: {_SCHEMA}")


_register()


def get_op() -> "torch._ops.OpOverloadPacket":
    """Return the registered ``warp_runner`` overload packet."""
    return getattr(getattr(torch.ops, NAMESPACE), OP_NAME)
