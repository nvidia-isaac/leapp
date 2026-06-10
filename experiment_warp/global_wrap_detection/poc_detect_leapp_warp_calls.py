#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Smoke test for array-interface-only Warp launch detection.

Run from the repo root with the Warp/APIC environment, for example:

    conda activate exp_env
    python experiment_warp/global_wrap_detection/poc_detect_leapp_warp_calls.py

This test does not patch any Warp function. It uses TracedWarpArray's
__array_interface__ / __cuda_array_interface__ hook to emit a torch.fx marker
node for Warp launches that consume traced arrays.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import FrameType
from typing import Any, Optional, Set, Tuple

import numpy as np
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import leapp
from leapp import annotate
from leapp.leapp_graph.datatypes.traced_warp_array import (
    leapp_warp_launch,
    set_array_interface_access_hook,
)


NODE_NAME = "warp_array_interface_detector"
DEFAULT_TRACE_DIR = Path(__file__).resolve().parent / "leapp_trace"
IMPORTED_LAUNCH_BEFORE_HOOK = wp.launch


@wp.kernel
def add_one_kernel(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i] = src[i] + 1.0


@wp.kernel
def tiled_add_kernel(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    i, tile = wp.tid()
    if tile == 0:
        dst[i] = src[i] + 2.0


@wp.kernel
def add_scaled_pair_kernel(
    left: wp.array(dtype=wp.float32),
    right: wp.array(dtype=wp.float32),
    scale: wp.float32,
    dst: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dst[i] = left[i] + right[i] * scale


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Warp device to use, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--trace-dir",
        default=str(DEFAULT_TRACE_DIR),
        help="Directory for LEAPP trace-time detection session files.",
    )
    return parser


def third_party_style_launch(src, dst, device: str) -> None:
    """Simulate a library that captured a launch alias before our hook exists."""

    IMPORTED_LAUNCH_BEFORE_HOOK(
        add_one_kernel,
        dim=src.size,
        inputs=[src],
        outputs=[dst],
        device=device,
    )


def _array_interface_hook(array, interface_name: str, caller_frame: Optional[FrameType]) -> None:
    pack_arg_frame = _find_frame(caller_frame, "warp._src.context", "pack_arg")
    if pack_arg_frame is None:
        return

    launch_frame = _find_frame(pack_arg_frame.f_back, "warp._src.context", "launch")
    tiled_frame = _find_frame(pack_arg_frame.f_back, "warp._src.context", "launch_tiled")

    print("\n[leapp-warp-array-interface] traced Warp array consumed by kernel arg packing")
    print(f"  interface: {interface_name}")
    print(f"  traced array: {_format_value(array)}")
    print(f"  pack_arg source: {_frame_label(pack_arg_frame)}")
    print("  pack_arg parameters:")
    _print_frame_values(pack_arg_frame, ("kernel", "arg_type", "arg_name", "value", "device", "adjoint"))

    if tiled_frame is not None:
        print(f"  outer entrypoint: warp.launch_tiled at {_frame_label(tiled_frame)}")
        print("  launch_tiled frame parameters:")
        _print_frame_values(tiled_frame, ("args", "kwargs", "device", "dim"))

    if launch_frame is not None:
        print(f"  launch dispatch: warp.launch at {_frame_label(launch_frame)}")
        print("  launch parameters:")
        _print_frame_values(
            launch_frame,
            (
                "kernel",
                "dim",
                "inputs",
                "outputs",
                "adj_inputs",
                "adj_outputs",
                "device",
                "stream",
                "adjoint",
                "record_tape",
                "record_cmd",
                "max_blocks",
                "block_dim",
            ),
        )


def _find_frame(frame: Optional[FrameType], module_name: str, function_name: str) -> Optional[FrameType]:
    while frame is not None:
        if frame.f_code.co_name == function_name and frame.f_globals.get("__name__") == module_name:
            return frame
        frame = frame.f_back
    return None


def _frame_label(frame: FrameType) -> str:
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def _print_frame_values(frame: FrameType, names: Tuple[str, ...]) -> None:
    for name in names:
        if name in frame.f_locals:
            print(f"    {name}={_format_value(frame.f_locals[name])}")


def _format_value(value: Any, seen: Optional[Set[int]] = None) -> str:
    if seen is None:
        seen = set()

    value_id = id(value)
    if value_id in seen:
        return "<recursive>"

    if isinstance(value, wp.array):
        pieces = [type(value).__name__]
        name = getattr(value, "name", None) or getattr(value, "_name", None)
        if name:
            pieces.append(f"name={name!r}")
        pieces.extend(
            [
                f"shape={value.shape}",
                f"dtype={value.dtype}",
                f"device={value.device}",
                f"ptr=0x{value.ptr:x}" if value.ptr else "ptr=None",
            ]
        )
        return "WarpArray(" + ", ".join(pieces) + ")"

    if _is_warp_kernel(value):
        return f"WarpKernel(key={getattr(value, 'key', None)!r})"

    if isinstance(value, dict):
        seen.add(value_id)
        inner = ", ".join(f"{key!r}: {_format_value(child, seen)}" for key, child in value.items())
        seen.remove(value_id)
        return "{" + inner + "}"

    if isinstance(value, (list, tuple)):
        seen.add(value_id)
        inner = ", ".join(_format_value(child, seen) for child in value)
        seen.remove(value_id)
        open_char, close_char = ("[", "]") if isinstance(value, list) else ("(", ")")
        return open_char + inner + close_char

    return repr(value)


def _is_warp_kernel(value: Any) -> bool:
    return value.__class__.__name__ == "Kernel" and hasattr(value, "key")


def _print_fx_warp_nodes(context) -> None:
    print("\nFX graph after Warp launch detection:")
    print(context.graph)

    warp_nodes = [
        node for node in context.graph.nodes
        if node.op == "call_function" and node.target is leapp_warp_launch
    ]
    print(f"\nRecorded leapp_warp_launch node count: {len(warp_nodes)}")
    for index, node in enumerate(warp_nodes):
        metadata = node.meta.get("leapp_warp_launch", {})
        traced_arrays_arg = node.args[1] if len(node.args) > 1 else ()
        print(f"\nFX Warp node {index}:")
        print(f"  name: {node.name}")
        print(f"  target: {node.target.__module__}.{node.target.__name__}")
        print(f"  proxy dependency count: {len(traced_arrays_arg)}")
        print(f"  metadata: {metadata}")


def run_smoke(device: str, trace_dir: str) -> None:
    wp.init()

    os.makedirs(trace_dir, exist_ok=True)
    leapp.start(
        name="warp_array_interface_detector_poc",
        save_path=trace_dir,
        global_patching=False,
        verbose=False,
    )

    previous_hook = None
    stopped = False
    try:
        src_np = np.arange(8, dtype=np.float32)
        other_np = np.linspace(1.0, 2.0, 8, dtype=np.float32)
        src_raw = wp.array(src_np, dtype=wp.float32, device=device)
        other_raw = wp.array(other_np, dtype=wp.float32, device=device)
        src, other = annotate.input_tensors(NODE_NAME, {"src": src_raw, "other": other_raw})

        launched = wp.empty_like(src_raw)
        tiled = wp.empty_like(src_raw)
        aliased = wp.empty_like(src_raw)
        mixed = wp.empty_like(src_raw)
        copied = wp.empty_like(src_raw)

        previous_hook = set_array_interface_access_hook(_array_interface_hook)

        print("\nCalling wp.launch with one LEAPP Warp array:")
        wp.launch(add_one_kernel, dim=src.size, inputs=[src], outputs=[launched], device=device)

        print("\nCalling wp.launch_tiled with one LEAPP Warp array:")
        wp.launch_tiled(
            tiled_add_kernel,
            dim=(src.size,),
            inputs=[src],
            outputs=[tiled],
            block_dim=4,
            device=device,
        )

        print("\nCalling a pre-imported launch alias from a third-party-style function:")
        third_party_style_launch(src, aliased, device)

        print("\nCalling wp.launch with two LEAPP Warp arrays and one scalar constant:")
        wp.launch(
            add_scaled_pair_kernel,
            dim=src.size,
            inputs=[src, other, 2.5],
            outputs=[mixed],
            device=device,
        )

        print("\nCalling non-kernel Warp ops; array-interface hook should stay quiet:")
        wp.copy(copied, src)
        src.zero_()
        src.fill_(3.0)

        _print_fx_warp_nodes(src.context_obj)

        # This POC is only about trace-time call detection. The current Warp output
        # path does not finalize raw/traced wp.array nodes yet, so stop the LEAPP
        # session without compiling.
        leapp.stop()
        stopped = True
    finally:
        set_array_interface_access_hook(previous_hook)
        if not stopped:
            leapp.stop()

    print(f"\nLEAPP detection session stopped: {Path(trace_dir) / 'warp_array_interface_detector_poc'}")


def main() -> None:
    args = _make_parser().parse_args()
    run_smoke(args.device, args.trace_dir)


if __name__ == "__main__":
    main()
