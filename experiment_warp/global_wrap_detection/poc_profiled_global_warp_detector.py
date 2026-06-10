#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Smoke test for global Warp function profiling without array-interface hooks.

Run from the repo root with the Warp/APIC environment, for example:

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate exp_env
    python -u experiment_warp/global_wrap_detection/poc_profiled_global_warp_detector.py --device cpu

This POC patches Python-visible Warp functions and selected Warp class methods.
It records calls that consume LEAPP-traced or detector-tracked Warp arrays and
satisfy one of the valid-effect rules: mutate a Warp array, return a Warp array,
or expose output Warp arrays in the function signature.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import leapp
from leapp import annotate

from experiment_warp.global_wrap_detection.warp_global_leapp_detector import (
    WarpLeappCallDetector,
)


NODE_NAME = "warp_profiled_global_detector"
DEFAULT_TRACE_DIR = Path(__file__).resolve().parent / "leapp_trace"
IMPORTED_LAUNCH_BEFORE_PATCH = wp.launch


def _make_closure_alias_runner():
    launch_alias = wp.launch

    def run(src, dst, device: str) -> None:
        launch_alias(add_one_kernel, dim=src.size, inputs=[src], outputs=[dst], device=device)

    return run


CLOSURE_LAUNCH_BEFORE_PATCH = _make_closure_alias_runner()


@wp.kernel
def add_one_kernel(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i] = src[i] + 1.0


@wp.kernel
def add_scaled_pair_kernel(
    left: wp.array(dtype=wp.float32),
    right: wp.array(dtype=wp.float32),
    scale: wp.float32,
    dst: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dst[i] = left[i] + right[i] * scale


@wp.kernel
def tiled_add_kernel(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    i, tile = wp.tid()
    if tile == 0:
        dst[i] = src[i] + 2.0


@wp.kernel
def sum_squares_kernel(src: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(loss, 0, src[i] * src[i])


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Warp device to use, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--trace-dir",
        default=str(DEFAULT_TRACE_DIR),
        help="Directory for LEAPP trace-time detection session files.",
    )
    parser.add_argument(
        "--verbose-ignored",
        action="store_true",
        help="Print ignored calls as the detector sees them.",
    )
    return parser


def _global_alias_launch(src, dst, device: str) -> None:
    IMPORTED_LAUNCH_BEFORE_PATCH(add_one_kernel, dim=src.size, inputs=[src], outputs=[dst], device=device)


def _scenario(detector: WarpLeappCallDetector, name: str, fn: Callable[[], None]) -> None:
    before_valid = len(detector.events)
    before_ignored = len(detector.ignored_events)
    print(f"\n=== {name} ===")
    try:
        fn()
    except Exception as exc:  # POC should keep exploring after unsupported cases.
        print(f"SCENARIO_ERROR {name}: {type(exc).__name__}: {exc}")
    new_valid = detector.events[before_valid:]
    new_ignored = detector.ignored_events[before_ignored:]
    print(f"SCENARIO_SUMMARY {name}: valid={len(new_valid)} ignored={len(new_ignored)}")
    for event in new_valid:
        print(f"  valid: {event.qualname} reasons={','.join(event.reasons)}")
    for event in new_ignored:
        print(f"  ignored: {event.qualname} reason={event.ignored_reason}")


def run_smoke(device: str, trace_dir: str, verbose_ignored: bool) -> None:
    wp.init()
    os.makedirs(trace_dir, exist_ok=True)

    leapp.start(
        name="warp_profiled_global_detector_poc",
        save_path=trace_dir,
        global_patching=False,
        verbose=False,
    )

    detector: WarpLeappCallDetector | None = None
    stopped = False
    try:
        src_np = np.arange(8, dtype=np.float32)
        other_np = np.linspace(1.0, 2.0, 8, dtype=np.float32)
        src_raw = wp.array(src_np, dtype=wp.float32, device=device)
        other_raw = wp.array(other_np, dtype=wp.float32, device=device)
        src, other = annotate.input_tensors(NODE_NAME, {"src": src_raw, "other": other_raw})

        direct_out = wp.empty_like(src_raw)
        chained_a = wp.empty_like(src_raw)
        chained_b = wp.empty_like(src_raw)
        tiled_out = wp.empty_like(src_raw)
        global_alias_out = wp.empty_like(src_raw)
        closure_alias_out = wp.empty_like(src_raw)
        copied = wp.empty_like(src_raw)
        cmd_out = wp.empty_like(src_raw)

        detector = WarpLeappCallDetector(verbose_ignored=verbose_ignored).install()
        print(f"patched_count={detector.patched_count}")

        _scenario(
            detector,
            "direct wp.launch with traced input and raw output",
            lambda: wp.launch(add_one_kernel, dim=src.size, inputs=[src], outputs=[direct_out], device=device),
        )

        def raw_intermediate_chain() -> None:
            wp.launch(add_one_kernel, dim=src.size, inputs=[src], outputs=[chained_a], device=device)
            wp.launch(
                add_scaled_pair_kernel,
                dim=src.size,
                inputs=[chained_a, other_raw, 3.0],
                outputs=[chained_b],
                device=device,
            )
            wp.launch(add_one_kernel, dim=src.size, inputs=[chained_b], outputs=[direct_out], device=device)

        _scenario(detector, "consecutive launches through raw intermediates", raw_intermediate_chain)

        _scenario(
            detector,
            "wp.launch_tiled",
            lambda: wp.launch_tiled(
                tiled_add_kernel,
                dim=(src.size,),
                inputs=[src],
                outputs=[tiled_out],
                block_dim=4,
                device=device,
            ),
        )

        _scenario(
            detector,
            "global alias imported before patch",
            lambda: _global_alias_launch(src, global_alias_out, device),
        )

        _scenario(
            detector,
            "closure alias imported before patch expected blind spot",
            lambda: CLOSURE_LAUNCH_BEFORE_PATCH(src, closure_alias_out, device),
        )

        _scenario(detector, "wp.copy traced source to raw dest", lambda: wp.copy(copied, src))
        _scenario(detector, "tracked raw dest zero_ mutator", lambda: copied.zero_())
        _scenario(detector, "traced input fill_ mutator", lambda: src.fill_(7.0))
        _scenario(detector, "traced input assign mutator", lambda: src.assign(other_raw))
        _scenario(detector, "wp.clone returns warp array", lambda: wp.clone(src))
        _scenario(detector, "wp.empty_like returns warp array", lambda: wp.empty_like(src))
        _scenario(detector, "wp.zeros_like returns and mutates warp array", lambda: wp.zeros_like(src))
        _scenario(detector, "wp.full_like returns and mutates warp array", lambda: wp.full_like(src, 5.0))
        _scenario(detector, "traced flatten returns warp array view", lambda: src.flatten())
        _scenario(detector, "wp.to_torch returns non-warp object", lambda: wp.to_torch(src))

        if str(device) == "cpu":
            _scenario(detector, "traced numpy returns non-warp object", lambda: src.numpy())

        def deferred_launch_command() -> None:
            cmd = wp.launch(
                add_one_kernel,
                dim=src.size,
                inputs=[src],
                outputs=[cmd_out],
                device=device,
                record_cmd=True,
            )
            cmd.launch()

        _scenario(detector, "record_cmd Launch.launch blind spot", deferred_launch_command)

        def tape_backward() -> None:
            # This is manually seeded as tracked because current TracedWarpArray does not
            # preserve requires_grad reliably enough for this autodiff smoke test.
            tape_src = wp.array(src_np, dtype=wp.float32, device=device, requires_grad=True)
            tape_loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
            detector.track_array(tape_src)
            with wp.Tape() as tape:
                wp.launch(sum_squares_kernel, dim=tape_src.size, inputs=[tape_src], outputs=[tape_loss], device=device)
            tape.backward(loss=tape_loss)

        _scenario(detector, "Tape.backward emits adjoint launch traffic", tape_backward)

        print("\n=== FINAL EVENT SUMMARY ===")
        print(f"valid_events={len(detector.events)} ignored_events={len(detector.ignored_events)}")
        by_name: dict[str, int] = {}
        for event in detector.events:
            by_name[event.qualname] = by_name.get(event.qualname, 0) + 1
        for qualname, count in sorted(by_name.items()):
            print(f"  {qualname}: {count}")

    finally:
        if detector is not None:
            detector.uninstall()
        if not stopped:
            leapp.stop()
            stopped = True


def main() -> None:
    args = _make_parser().parse_args()
    run_smoke(args.device, args.trace_dir, args.verbose_ignored)


if __name__ == "__main__":
    main()
