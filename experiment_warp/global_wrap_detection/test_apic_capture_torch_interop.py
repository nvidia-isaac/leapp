"""Smoke test APIC capture across unrelated CUDA torch work.

The captured region intentionally has this shape:

    wp.ScopedCapture()
        wp.launch(k1)
        wp.launch(k2)
        torch_function(unrelated_cuda_torch_tensor)
        wp.launch(k3)

The assertion is that capture/save/load/replay does not crash and that the
captured graph's named Warp outputs are equivalent to replaying only k1, k2,
and k3.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate exp_env
    python experiment_warp/global_wrap_detection/test_apic_capture_torch_interop.py
"""

from __future__ import annotations

import shutil
import tempfile
import os
from pathlib import Path

import numpy as np

try:
    import warp as wp
except ModuleNotFoundError:  # pragma: no cover - optional experiment dependency
    wp = None


DEVICE = "cuda:0"
N = 16


if wp is not None:

    @wp.kernel
    def k1(a: wp.array(dtype=float), x: wp.array(dtype=float)):
        i = wp.tid()
        x[i] = a[i] + 1.0

    @wp.kernel
    def k2(x: wp.array(dtype=float), y: wp.array(dtype=float)):
        i = wp.tid()
        y[i] = x[i] * 2.0

    @wp.kernel
    def k3(y: wp.array(dtype=float), out: wp.array(dtype=float)):
        i = wp.tid()
        out[i] = y[i] - 3.0


def _skip(reason: str) -> None:
    if "PYTEST_CURRENT_TEST" not in os.environ:
        print(f"SKIP: {reason}")
        raise SystemExit(0)

    try:
        import pytest  # noqa: PLC0415
    except ModuleNotFoundError:
        print(f"SKIP: {reason}")
        raise SystemExit(0)
    pytest.skip(reason)


def _require_warp_cuda():
    if wp is None:
        _skip("warp is not installed")

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        _skip("torch CUDA is not available")

    wp.init()
    if hasattr(wp, "is_cuda_available") and not wp.is_cuda_available():
        _skip("warp CUDA is not available")

    return torch


def _torch_function(unrelated):
    unrelated.mul_(1.25)
    unrelated.add_(0.5)
    return unrelated


def _run_warp_only(a_np: np.ndarray) -> np.ndarray:
    a = wp.array(a_np, dtype=float, device=DEVICE)
    x = wp.zeros(N, dtype=float, device=DEVICE)
    y = wp.zeros(N, dtype=float, device=DEVICE)
    out = wp.zeros(N, dtype=float, device=DEVICE)

    wp.launch(k1, dim=N, inputs=[a], outputs=[x], device=DEVICE)
    wp.launch(k2, dim=N, inputs=[x], outputs=[y], device=DEVICE)
    wp.launch(k3, dim=N, inputs=[y], outputs=[out], device=DEVICE)
    wp.synchronize_device(DEVICE)
    return out.numpy().copy()


def _run_captured_with_unrelated_torch(a_np: np.ndarray, torch) -> np.ndarray:
    workdir = Path(tempfile.mkdtemp(prefix="apic_torch_interop_"))
    try:
        a = wp.array(a_np, dtype=float, device=DEVICE)
        x = wp.zeros(N, dtype=float, device=DEVICE)
        y = wp.zeros(N, dtype=float, device=DEVICE)
        out = wp.zeros(N, dtype=float, device=DEVICE)
        unrelated_tensor = torch.arange(32, dtype=torch.float32, device=DEVICE)

        capture_path = str(workdir / "captured")
        with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as capture:
            wp.launch(k1, dim=N, inputs=[a], outputs=[x], device=DEVICE)
            wp.launch(k2, dim=N, inputs=[x], outputs=[y], device=DEVICE)
            unrelated_tensor.mul_(1.25)
            unrelated_tensor.add_(0.5)
            wp.launch(k3, dim=N, inputs=[y], outputs=[out], device=DEVICE)

        wp.capture_save(capture.graph, capture_path, inputs={"a": a}, outputs={"out": out})

        replay_a = wp.array(a_np, dtype=float, device=DEVICE)
        replay_out = wp.zeros(N, dtype=float, device=DEVICE)
        loaded = wp.capture_load(capture_path, device=DEVICE)
        loaded.set_param("a", replay_a)
        loaded.set_param("out", replay_out)
        wp.capture_launch(loaded)
        wp.synchronize_device(DEVICE)

        result = wp.empty(N, dtype=float, device=DEVICE)
        loaded.get_param("out", result)
        return result.numpy().copy()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_apic_capture_ignores_unrelated_torch_cuda_work() -> None:
    torch = _require_warp_cuda()
    a_np = np.linspace(-2.0, 2.0, N, dtype=np.float32)

    expected = _run_warp_only(a_np)
    captured = _run_captured_with_unrelated_torch(a_np, torch)

    np.testing.assert_allclose(captured, expected, rtol=1e-6, atol=1e-6)


def main() -> None:
    test_apic_capture_ignores_unrelated_torch_cuda_work()
    print("PASS: APIC capture with unrelated CUDA torch work matches k1/k2/k3 replay")


if __name__ == "__main__":
    main()
