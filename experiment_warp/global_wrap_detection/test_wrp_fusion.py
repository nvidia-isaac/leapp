"""Empirically test whether N .wrp graphs can be fused into a single .wrp.

Two strategies are compared:

  Strategy A (the literal question):
    Save N single-launch .wrp files, load all N, then `capture_launch` each
    loaded graph inside ONE outer `wp.ScopedCapture(apic=True)` and
    `capture_save` the outer graph. Reload the fused .wrp and check it applies
    all N operations.

  Strategy B (re-issue the launches):
    Inside ONE outer `wp.ScopedCapture(apic=True)`, call the actual `wp.launch`
    N times, then `capture_save`. Reload and check.

Each unit op is `add_one`: x[i] += 1. So a correctly fused graph over N=10
units must turn a zero buffer into all-10.0 after one fused replay.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate exp_env
    python experiment_warp/global_wrap_detection/test_wrp_fusion.py
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import warp as wp


DEVICE = "cuda:0"
N = 10
SIZE = 8


@wp.kernel
def add_one(x: wp.array(dtype=float)):
    i = wp.tid()
    x[i] = x[i] + 1.0


def _make_unit_wrp(path: str) -> None:
    """Capture a single add_one launch and save it as one .wrp."""
    x = wp.zeros(SIZE, dtype=float, device=DEVICE)
    with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as cap:
        wp.launch(add_one, dim=SIZE, inputs=[x], device=DEVICE)
    wp.capture_save(cap.graph, path, inputs={"x": x}, outputs={"x": x})


def strategy_a_replay_loaded(workdir: str) -> tuple[bool, str]:
    """Load N .wrp and capture_launch them inside one outer APIC capture."""
    unit_paths = [os.path.join(workdir, f"unit_{i}") for i in range(N)]
    for p in unit_paths:
        _make_unit_wrp(p)

    shared = wp.zeros(SIZE, dtype=float, device=DEVICE)
    loaded = [wp.capture_load(p, device=DEVICE) for p in unit_paths]
    for g in loaded:
        g.set_param("x", shared)

    fused_path = os.path.join(workdir, "fused_a")
    try:
        with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as outer:
            for g in loaded:
                wp.capture_launch(g)
        wp.capture_save(outer.graph, fused_path, inputs={"x": shared}, outputs={"x": shared})
    except Exception as exc:  # noqa: BLE001
        return False, f"capture/save raised: {type(exc).__name__}: {exc}"

    # Reload the fused graph into a fresh buffer and replay once.
    try:
        result = wp.zeros(SIZE, dtype=float, device=DEVICE)
        fused = wp.capture_load(fused_path, device=DEVICE)
        fused.set_param("x", result)
        wp.capture_launch(fused)
        wp.synchronize_device(DEVICE)
        out = wp.empty(SIZE, dtype=float, device=DEVICE)
        fused.get_param("x", out)
        vals = out.numpy()
    except Exception as exc:  # noqa: BLE001
        return False, f"reload/replay raised: {type(exc).__name__}: {exc}"

    ok = bool(np.allclose(vals, float(N)))
    return ok, f"fused replay -> {vals.tolist()} (expected all {float(N)})"


def strategy_b_relaunch(workdir: str) -> tuple[bool, str]:
    """Re-issue the actual wp.launch N times inside one outer APIC capture."""
    x = wp.zeros(SIZE, dtype=float, device=DEVICE)
    fused_path = os.path.join(workdir, "fused_b")
    try:
        with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as outer:
            for _ in range(N):
                wp.launch(add_one, dim=SIZE, inputs=[x], device=DEVICE)
        wp.capture_save(outer.graph, fused_path, inputs={"x": x}, outputs={"x": x})
    except Exception as exc:  # noqa: BLE001
        return False, f"capture/save raised: {type(exc).__name__}: {exc}"

    try:
        result = wp.zeros(SIZE, dtype=float, device=DEVICE)
        fused = wp.capture_load(fused_path, device=DEVICE)
        fused.set_param("x", result)
        wp.capture_launch(fused)
        wp.synchronize_device(DEVICE)
        out = wp.empty(SIZE, dtype=float, device=DEVICE)
        fused.get_param("x", out)
        vals = out.numpy()
    except Exception as exc:  # noqa: BLE001
        return False, f"reload/replay raised: {type(exc).__name__}: {exc}"

    ok = bool(np.allclose(vals, float(N)))
    return ok, f"fused replay -> {vals.tolist()} (expected all {float(N)})"


def main() -> None:
    wp.init()
    workdir = tempfile.mkdtemp(prefix="wrp_fusion_")
    try:
        a_ok, a_msg = strategy_a_replay_loaded(workdir)
        b_ok, b_msg = strategy_b_relaunch(workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("=" * 70)
    print(f"Strategy A (load N .wrp, capture_launch inside outer capture): "
          f"{'PASS' if a_ok else 'FAIL'}")
    print(f"    {a_msg}")
    print(f"Strategy B (re-issue N wp.launch inside outer capture):        "
          f"{'PASS' if b_ok else 'FAIL'}")
    print(f"    {b_msg}")
    print("=" * 70)


if __name__ == "__main__":
    main()
