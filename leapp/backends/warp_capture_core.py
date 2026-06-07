#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Reusable Warp capture helpers shared by the standalone warp_node() and WarpRegionNode.

A "record" is a dict with at least {"args": tuple, "kwargs": dict} captured from a patched
wp.launch call. Replay runs ONLY the recorded launches inside one wp.ScopedCapture(apic=True);
the arrays were already allocated during the eager pass, so nothing allocates during capture
(allocating inside a CUDA-graph capture segfaults).
"""


def resolve_device(wp, records, explicit):
    """Determine the CUDA device to use for the APIC capture.

    Checks, in order:
      1. The explicit override (if provided).
      2. A top-level ``"device"`` key in any record (shape stored by warp_capture.py).
      3. The ``"device"`` key inside ``record["kwargs"]`` (shape used by callers that build
         records directly without the standalone patched_launch).
      4. Any ``wp.array`` found in top-level ``"inputs"``/``"outputs"`` keys.
      5. Any ``wp.array`` found in ``kwargs["inputs"]``/``kwargs["outputs"]``.
      6. Falls back to ``"cuda:0"``.
    """
    if explicit:
        return str(explicit)
    # Pass 1: check top-level "device" key (standalone warp_capture.py shape)
    for r in records:
        dev = r.get("device")
        if dev is not None:
            return str(dev)
    # Pass 2: check kwargs["device"] (caller-built record shape)
    for r in records:
        dev = r["kwargs"].get("device")
        if dev is not None:
            return str(dev)
    # Pass 3: check top-level "inputs"/"outputs" for wp.array
    for r in records:
        ins = r.get("inputs") or []
        outs = r.get("outputs") or []
        for a in list(ins) + list(outs):
            if isinstance(a, wp.array):
                return str(a.device)
    # Pass 4: check kwargs["inputs"]/"outputs" for wp.array
    for r in records:
        ins = r["kwargs"].get("inputs") or []
        outs = r["kwargs"].get("outputs") or []
        for a in list(ins) + list(outs):
            if isinstance(a, wp.array):
                return str(a.device)
    return "cuda:0"


def replay_into_apic_capture(wp, records, device, orig_launch=None):
    """Replay all recorded wp.launch calls inside one wp.ScopedCapture(apic=True).

    Args:
        wp: The warp module.
        records: List of dicts, each with at least ``{"args": tuple, "kwargs": dict}``.
        device: The CUDA device string (e.g. ``"cuda:0"``) to capture on.
        orig_launch: The original (unpatched) ``wp.launch`` to use for replay. If ``None``,
            ``wp.launch`` is used as-is (safe when called outside a patched context).

    Returns:
        The captured ``wp.Graph`` (``cap.graph``).
    """
    launch = orig_launch if orig_launch is not None else wp.launch
    with wp.ScopedCapture(device=device, force_module_load=True, apic=True) as cap:
        for r in records:
            launch(*r["args"], **r["kwargs"])
    return cap.graph
