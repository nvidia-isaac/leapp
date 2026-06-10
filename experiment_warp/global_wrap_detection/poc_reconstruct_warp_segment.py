#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""POC: reconstruct consecutive Warp launches from traced-array detection.

This intentionally does not keep APIC capture open during the eager user pass.
It records live wp.Kernel launch recipes when TracedWarpArray trips Warp's
argument packing path, then replays the collected recipes inside one
wp.ScopedCapture(apic=True) to produce one fused .wrp.

The script runs two cases:

1. reconstructible_chain:
   Three consecutive launches are all visible because each launch consumes at
   least one LEAPP-traced array. The recorded recipes replay into one .wrp and
   the loaded .wrp matches the eager result.

2. raw_intermediate_blind_spot:
   Only the first launch consumes LEAPP-traced arrays. Later launches consume
   raw intermediate wp.array buffers, so the traced-array hook does not fire.
   This demonstrates the current limit of using array-interface detection alone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
import shutil
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
from leapp.leapp_graph.datatypes.traced_warp_array import set_array_interface_access_hook


N = 8
VISIBLE_NODE = "warp_segment_reconstruct_visible"
BLIND_NODE = "warp_segment_reconstruct_blind_spot"
DEFAULT_TRACE_DIR = Path(__file__).resolve().parent / "leapp_trace" / "warp_segment_reconstruct_poc"


@wp.kernel
def add_fields(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    summed: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    summed[i] = a[i] + b[i]


@wp.kernel
def scale_with_bias_anchor(
    summed: wp.array(dtype=wp.float32),
    bias: wp.array(dtype=wp.float32),
    anchor: wp.array(dtype=wp.float32),
    scaled: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    scaled[i] = summed[i] * 2.0 + bias[i] + anchor[i] * 0.0


@wp.kernel
def finalize_anchor(
    scaled: wp.array(dtype=wp.float32),
    anchor: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = scaled[i] * 0.5 + anchor[i] * 0.0


@wp.kernel
def scale_with_bias(
    summed: wp.array(dtype=wp.float32),
    bias: wp.array(dtype=wp.float32),
    scaled: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    scaled[i] = summed[i] * 2.0 + bias[i]


@wp.kernel
def finalize(
    scaled: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = scaled[i] * 0.5


@dataclass
class WarpLaunchRecipe:
    index: int
    op: str
    kernel: object
    dim: object
    inputs: list
    outputs: list
    kwargs: dict
    source: str
    traced_arg_names: tuple[str, ...]

    def replay(self) -> None:
        if self.op != "wp.launch":
            raise NotImplementedError(f"Replay for {self.op} is not implemented in this POC.")
        wp.launch(
            self.kernel,
            dim=self.dim,
            inputs=self.inputs,
            outputs=self.outputs,
            **self.kwargs,
        )


@dataclass
class WarpSegmentRecipe:
    segment_id: str
    node_name: str
    launches: list[WarpLaunchRecipe] = field(default_factory=list)
    _seen_launch_keys: Set[Tuple[int, int, int]] = field(default_factory=set)

    def add_launch(self, launch_frame: FrameType, traced_arg_names: tuple[str, ...]) -> bool:
        key = (id(launch_frame), id(launch_frame.f_code), launch_frame.f_lineno)
        if key in self._seen_launch_keys:
            return False

        self._seen_launch_keys.add(key)
        kwargs = {"device": launch_frame.f_locals.get("device")}
        for name in ("stream", "adjoint", "record_tape", "max_blocks", "block_dim"):
            value = launch_frame.f_locals.get(name)
            if value is not None:
                kwargs[name] = value

        self.launches.append(
            WarpLaunchRecipe(
                index=len(self.launches),
                op="wp.launch",
                kernel=launch_frame.f_locals["kernel"],
                dim=launch_frame.f_locals.get("dim"),
                inputs=list(launch_frame.f_locals.get("inputs") or []),
                outputs=list(launch_frame.f_locals.get("outputs") or []),
                kwargs=kwargs,
                source=f"{launch_frame.f_code.co_filename}:{launch_frame.f_lineno}",
                traced_arg_names=traced_arg_names,
            )
        )
        return True

    def replay(self) -> None:
        # Replay must use a snapshot. If the trace-time recorder is accidentally
        # left active, iterating the live list can append forever.
        for launch in list(self.launches):
            launch.replay()


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Warp device to use. APIC .wrp replay is tested on CUDA.")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR), help="Directory for generated POC artifacts.")
    return parser


def _segment_hook(array, interface_name: str, caller_frame: Optional[FrameType]) -> None:
    del interface_name
    pack_arg_frame = _find_frame(caller_frame, "warp._src.context", "pack_arg")
    if pack_arg_frame is None:
        return

    launch_frame = _find_frame(pack_arg_frame.f_back, "warp._src.context", "launch")
    if launch_frame is None:
        return

    context = array.context_obj
    segment = getattr(context, "_poc_warp_segment_recipe", None)
    if segment is None:
        segment = WarpSegmentRecipe(
            segment_id=f"{context.name}_segment_0",
            node_name=context.name,
        )
        setattr(context, "_poc_warp_segment_recipe", segment)

    traced_names = _traced_arg_names(
        list(launch_frame.f_locals.get("inputs") or []),
        list(launch_frame.f_locals.get("outputs") or []),
    )
    added = segment.add_launch(launch_frame, traced_names)
    if added:
        launch = segment.launches[-1]
        print(
            f"[segment-recorder] recorded launch {launch.index}: "
            f"{_kernel_label(launch.kernel)} traced_args={launch.traced_arg_names}"
        )


def _find_frame(frame: Optional[FrameType], module_name: str, function_name: str) -> Optional[FrameType]:
    while frame is not None:
        if frame.f_code.co_name == function_name and frame.f_globals.get("__name__") == module_name:
            return frame
        frame = frame.f_back
    return None


def _traced_arg_names(inputs: list, outputs: list) -> tuple[str, ...]:
    names = []
    for value in [*inputs, *outputs]:
        context = getattr(value, "context_obj", None)
        if context is not None:
            names.append(getattr(value, "name", "<unnamed>"))
    return tuple(names)


def _kernel_label(kernel: object) -> str:
    func = getattr(kernel, "func", None)
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(kernel)


def _expected(a_np: np.ndarray, b_np: np.ndarray, bias_np: np.ndarray) -> np.ndarray:
    return ((a_np + b_np) * 2.0 + bias_np) * 0.5


def _make_arrays(device: str):
    a_np = np.arange(N, dtype=np.float32)
    b_np = np.full(N, 10.0, dtype=np.float32)
    bias_np = np.linspace(0.0, 1.0, N, dtype=np.float32)

    arrays = {
        "a_raw": wp.array(a_np, dtype=wp.float32, device=device),
        "b_raw": wp.array(b_np, dtype=wp.float32, device=device),
        "bias": wp.array(bias_np, dtype=wp.float32, device=device),
        "summed": wp.empty(N, dtype=wp.float32, device=device),
        "scaled": wp.empty(N, dtype=wp.float32, device=device),
        "out": wp.empty(N, dtype=wp.float32, device=device),
    }
    return a_np, b_np, bias_np, arrays


def _reset_arrays(arrays: dict[str, Any], a_np: np.ndarray, b_np: np.ndarray, bias_np: np.ndarray) -> None:
    arrays["a_raw"].assign(a_np)
    arrays["b_raw"].assign(b_np)
    arrays["bias"].assign(bias_np)
    arrays["summed"].zero_()
    arrays["scaled"].zero_()
    arrays["out"].zero_()


def _run_visible_chain(a, b, arrays: dict[str, Any], device: str) -> None:
    wp.launch(add_fields, dim=N, inputs=[a, b], outputs=[arrays["summed"]], device=device)
    print("  eager Python between launches is legal because APIC capture is not open")
    wp.launch(
        scale_with_bias_anchor,
        dim=N,
        inputs=[arrays["summed"], arrays["bias"], a],
        outputs=[arrays["scaled"]],
        device=device,
    )
    wp.launch(
        finalize_anchor,
        dim=N,
        inputs=[arrays["scaled"], b],
        outputs=[arrays["out"]],
        device=device,
    )


def _run_blind_spot_chain(a, b, arrays: dict[str, Any], device: str) -> None:
    wp.launch(add_fields, dim=N, inputs=[a, b], outputs=[arrays["summed"]], device=device)
    wp.launch(
        scale_with_bias,
        dim=N,
        inputs=[arrays["summed"], arrays["bias"]],
        outputs=[arrays["scaled"]],
        device=device,
    )
    wp.launch(finalize, dim=N, inputs=[arrays["scaled"]], outputs=[arrays["out"]], device=device)


def _print_segment(segment: Optional[WarpSegmentRecipe]) -> None:
    if segment is None:
        print("  recorded segment: <none>")
        return

    print(f"  recorded segment: {segment.segment_id}")
    print(f"  launch count: {len(segment.launches)}")
    for launch in segment.launches:
        print(
            f"    {launch.index}: {_kernel_label(launch.kernel)} "
            f"dim={launch.dim} inputs={len(launch.inputs)} outputs={len(launch.outputs)} "
            f"traced_args={launch.traced_arg_names}"
        )


def _capture_replayed_segment(
    segment: WarpSegmentRecipe,
    arrays: dict[str, Any],
    a,
    b,
    output_base: Path,
    device: str,
) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_base.with_suffix(".wrp").unlink(missing_ok=True)
    shutil.rmtree(output_base.with_name(f"{output_base.name}_modules"), ignore_errors=True)

    previous_hook = set_array_interface_access_hook(None)
    try:
        with wp.ScopedCapture(device=device, force_module_load=True, apic=True) as capture:
            segment.replay()
    finally:
        set_array_interface_access_hook(previous_hook)

    wp.capture_save(
        capture.graph,
        str(output_base),
        inputs={"a": a, "b": b},
        outputs={"out": arrays["out"]},
    )


def _verify_loaded_wrp(
    output_base: Path,
    a_np: np.ndarray,
    b_np: np.ndarray,
    expected: np.ndarray,
    device: str,
) -> tuple[bool, np.ndarray, list[str]]:
    a2 = wp.array(a_np, dtype=wp.float32, device=device)
    b2 = wp.array(b_np, dtype=wp.float32, device=device)
    loaded = wp.capture_load(str(output_base), device=device)
    loaded.set_param("a", a2)
    loaded.set_param("b", b2)
    wp.capture_launch(loaded)
    wp.synchronize_device(device)

    out = wp.empty(N, dtype=wp.float32, device=device)
    loaded.get_param("out", out)
    loaded_result = out.numpy()
    params = list(getattr(loaded, "params", []) or [])
    return bool(np.allclose(loaded_result, expected)), loaded_result, params


def run_visible_case(device: str, trace_dir: Path) -> None:
    print("\n=== reconstructible_chain ===")
    leapp.start(
        name="warp_segment_reconstruct_visible_poc",
        save_path=str(trace_dir),
        global_patching=False,
        verbose=False,
    )
    previous_hook = None
    stopped = False
    try:
        a_np, b_np, bias_np, arrays = _make_arrays(device)
        a, b = annotate.input_tensors(VISIBLE_NODE, {"a": arrays["a_raw"], "b": arrays["b_raw"]})

        previous_hook = set_array_interface_access_hook(_segment_hook)
        _run_visible_chain(a, b, arrays, device)
        wp.synchronize_device(device)

        eager_result = arrays["out"].numpy().copy()
        expected = _expected(a_np, b_np, bias_np)
        eager_ok = bool(np.allclose(eager_result, expected))

        segment = getattr(a.context_obj, "_poc_warp_segment_recipe", None)
        _print_segment(segment)
        if segment is None or len(segment.launches) != 3:
            raise RuntimeError("Expected to reconstruct three launches for the visible chain.")

        _reset_arrays(arrays, a_np, b_np, bias_np)
        output_base = trace_dir / "reconstructed_visible_segment"
        _capture_replayed_segment(segment, arrays, a, b, output_base, device)

        arrays["bias"].assign(np.full(N, 999.0, dtype=np.float32))
        loaded_ok, loaded_result, params = _verify_loaded_wrp(output_base, a_np, b_np, expected, device)
        param_ok = "a" in params and "b" in params and "out" in params and "bias" not in params

        print(f"  eager matches expected: {'PASS' if eager_ok else 'FAIL'}")
        print(f"  replayed one-.wrp matches expected: {'PASS' if loaded_ok else 'FAIL'}")
        print(f"  baked bias is not a runtime param: {'PASS' if param_ok else 'FAIL'}")
        print(f"  wrote: {output_base.with_suffix('.wrp')}")
        print(f"  loaded.params = {params}")
        print(f"  loaded out = {loaded_result.tolist()}")

        leapp.stop()
        stopped = True
    finally:
        set_array_interface_access_hook(previous_hook)
        if not stopped:
            leapp.stop()


def run_blind_spot_case(device: str, trace_dir: Path) -> None:
    print("\n=== raw_intermediate_blind_spot ===")
    leapp.start(
        name="warp_segment_reconstruct_blind_spot_poc",
        save_path=str(trace_dir),
        global_patching=False,
        verbose=False,
    )
    previous_hook = None
    stopped = False
    try:
        _, _, _, arrays = _make_arrays(device)
        a, b = annotate.input_tensors(BLIND_NODE, {"a": arrays["a_raw"], "b": arrays["b_raw"]})

        previous_hook = set_array_interface_access_hook(_segment_hook)
        _run_blind_spot_chain(a, b, arrays, device)
        wp.synchronize_device(device)

        segment = getattr(a.context_obj, "_poc_warp_segment_recipe", None)
        _print_segment(segment)
        recorded = 0 if segment is None else len(segment.launches)
        print(f"  expected current hook visibility: 1 of 3 launches")
        print(f"  observed current hook visibility: {recorded} of 3 launches")
        print("  conclusion: raw intermediates need output propagation or a broader Warp command recorder")

        leapp.stop()
        stopped = True
    finally:
        set_array_interface_access_hook(previous_hook)
        if not stopped:
            leapp.stop()


def main() -> None:
    args = _make_parser().parse_args()
    wp.init()
    trace_dir = Path(args.trace_dir)
    os.makedirs(trace_dir, exist_ok=True)
    run_visible_case(args.device, trace_dir)
    run_blind_spot_case(args.device, trace_dir)


if __name__ == "__main__":
    main()
