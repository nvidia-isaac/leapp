#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Internal diagnostic messages and interpretation for export and validation.

Helpers return explanatory text from exceptions or recorded validation results.
Callers own model execution, logging, and exception raising.
"""

import os

import torch

_INFERENCE_TENSOR_ERRORS = (
    "Inference tensors cannot be saved for backward",
    "Setting requires_grad=True on inference tensor outside InferenceMode",
)
_INFERENCE_TENSOR_HINT = (
    "LEAPP could not export node '{node_name}' with backend '{backend}' because "
    "PyTorch rejected an inference tensor. This may be caused by "
    "torch.inference_mode() during model initialization or capture, including "
    "decorated functions. Use torch.no_grad() instead in those paths. Recreate "
    "affected tensors outside inference mode; leaving inference mode does not "
    "convert existing tensors."
)
_REENTRY_VALIDATION_HINT = (
    "Sample 0 (initial trace) passed validation, but one or more re-entry "
    "samples failed. This pattern often means a value that changes across "
    "iterations was inlined as a constant during export. Consider declaring "
    "it with annotate.input_tensors() for this node."
)
_HIDDEN_STATE_HINT = (
    "Sample 0 passed validation, but replaying it after the other samples gave a "
    "different result. This can indicate that the exported model keeps state "
    "between calls that was not declared, or that execution is nondeterministic. "
    "Undeclared state usually means a buffer allocated outside the node is "
    "reset and then accumulated into inside it, for example a Warp array cleared "
    "with zero_() and written with wp.atomic_add. Declare such buffers with "
    "annotate.register_buffer() for this node, or allocate them inside the node. "
    "Declare state that should carry over between calls with "
    "annotate.state_tensors() and annotate.update_state()."
)
_TF32_HINT = (
    "On an NVIDIA GPU, numerical differences can come from TF32 settings that "
    "differ between capture and validation. If TF32 may be involved, set "
    "NVIDIA_TF32_OVERRIDE=0 before starting Python, then repeat capture, export, "
    "and validation."
)


def mismatch_type(
    exported: torch.Tensor, source: torch.Tensor, *, rtol: float, atol: float
) -> str:
    """Return "tolerance" for a miss that rounding can explain, "integer" for a
    miss in an integer output or a larger miss between whole numbers in a float
    output, otherwise "value".

    The caller has already checked that the shapes match, that neither tensor
    holds NaN or Inf, and that some elements miss the tolerance.
    """
    if not source.dtype.is_floating_point:
        return "integer"
    # A wider share would also take in undeclared state that moves an output by
    # a few percent; bfloat16 rounds 8 times more coarsely than float16.
    share = 0.03 if source.dtype == torch.bfloat16 else 0.01
    missed = ~torch.isclose(exported, source, rtol=rtol, atol=atol)
    missed_diff = (exported - source).abs()[missed].max().item()
    # Only missed elements count: large values move the most and can still pass.
    # The second bound keeps out an error that is large next to the values that miss.
    small_share = missed_diff <= share * source.abs().max().item()
    small_near_zero = missed_diff <= max(
        0.1 * source[missed].abs().max().item(), 10 * atol
    )
    if small_share and small_near_zero:
        return "tolerance"
    # Rounding can flip a flag or an index stored as a float, such as
    # (x > 0).float(), like an integer output. From 1 / eps on, every value of
    # the dtype is a whole number, so a whole number says nothing there.
    values = torch.cat((exported[missed], source[missed]))
    whole = torch.equal(values, values.round())
    limit = 1 / torch.finfo(source.dtype).eps
    return "integer" if whole and values.abs().max().item() < limit else "value"


def export_error_hint(error: Exception, *, node_name: str, backend: str) -> str | None:
    """Recognize inference-tensor failures through an exporter's exception chain."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if any(message in str(error) for message in _INFERENCE_TENSOR_ERRORS):
            return _INFERENCE_TENSOR_HINT.format(node_name=node_name, backend=backend)
        error = error.__cause__ or error.__context__
    return None


def validation_hint(
    sample_passed: dict[int, bool],
    *,
    replay_mismatches: set[str],
    mismatches: set[str],
) -> str | None:
    """Explain a replay failure first, otherwise a failure confined to re-entry.

    Then add the TF32 hint, alone or after the re-entry hint, if every float
    miss is small ("tolerance"), the machine has an NVIDIA GPU, and
    NVIDIA_TF32_OVERRIDE is not already 0. A miss in an integer output does not
    count against it, since rounding can flip one, as with argmax.
    """
    if replay_mismatches:
        return _HIDDEN_STATE_HINT
    hints = []
    if sample_passed.get(0) and any(
        not passed for index, passed in sample_passed.items() if index > 0
    ):
        hints.append(_REENTRY_VALIDATION_HINT)
    if (
        mismatches - {"integer"} == {"tolerance"}
        and torch.cuda.is_available()
        and os.environ.get("NVIDIA_TF32_OVERRIDE") != "0"
    ):
        hints.append(_TF32_HINT)
    return " ".join(hints) or None
