#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Internal diagnostic messages and interpretation for export and validation.

Helpers return explanatory text from exceptions or recorded validation results.
Callers own model execution, logging, and exception raising.
"""

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

    Add the TF32 hint when the samples' only ``mismatches`` are values beyond the
    tolerance and ``replay_mismatches`` add no other type. The replay's own
    tolerance miss does not count, since the hidden-state hint explains it.
    """
    hints = []
    if replay_mismatches:
        hints.append(_HIDDEN_STATE_HINT)
    elif sample_passed.get(0) and any(
        not passed for index, passed in sample_passed.items() if index > 0
    ):
        hints.append(_REENTRY_VALIDATION_HINT)
    if mismatches | (replay_mismatches - {"tolerance"}) == {"tolerance"}:
        hints.append(_TF32_HINT)
    return " ".join(hints) or None
