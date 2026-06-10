"""Prove the command-log / replay model for fusing a Warp op sequence.

Models the proposed LEAPP design:

  1. EAGER PASS: run a chain of Warp launches eagerly while recording each one
     into an execution-sequence ledger. Eager execution means the user could
     freely print / .numpy() between launches (no capture is open here).
       - tracked arrays  : a, b (inputs), averaged (output)  -> become .wrp params
       - non-tracked array: bias (a constant used mid-chain)  -> must be BAKED
  2. RESET + CONTROLLED REPLAY: restore tracked inputs to their stored initial
     values, then re-issue the logged launches inside ONE ScopedCapture(apic=True)
     and capture_save naming ONLY the tracked arrays.
  3. VERIFY: load the fused .wrp into fresh buffers, set_param only the tracked
     inputs, launch once, read the tracked output, compare to the eager result.

Key things this proves:
  - A multi-kernel chain over distinct buffers fuses into a single valid .wrp.
  - A non-tracked array (bias) is frozen into the .wrp as a constant: the loaded
    graph reproduces the correct result even though bias is never set_param'd,
    and mutating the original bias after capture does not change the result.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate exp_env
    python experiment_warp/global_wrap_detection/test_command_log_replay.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import warp as wp


DEVICE = "cuda:0"
N = 8


@wp.kernel
def add_fields(a: wp.array(dtype=float), b: wp.array(dtype=float), summed: wp.array(dtype=float)):
    i = wp.tid()
    summed[i] = a[i] + b[i]


@wp.kernel
def scale_with_bias(
    summed: wp.array(dtype=float),
    bias: wp.array(dtype=float),  # non-tracked constant
    scaled: wp.array(dtype=float),
):
    i = wp.tid()
    scaled[i] = summed[i] * 2.0 + bias[i]


@wp.kernel
def finalize(scaled: wp.array(dtype=float), averaged: wp.array(dtype=float)):
    i = wp.tid()
    averaged[i] = scaled[i] * 0.5


def main() -> None:
    wp.init()
    workdir = Path(tempfile.mkdtemp(prefix="cmd_log_replay_"))
    try:
        a_np = np.arange(N, dtype=np.float32)
        b_np = np.full(N, 10.0, dtype=np.float32)
        bias_np = np.linspace(0.0, 1.0, N, dtype=np.float32)  # the baked constant

        a = wp.array(a_np, dtype=float, device=DEVICE)
        b = wp.array(b_np, dtype=float, device=DEVICE)
        bias = wp.array(bias_np, dtype=float, device=DEVICE)
        summed = wp.zeros(N, dtype=float, device=DEVICE)
        scaled = wp.zeros(N, dtype=float, device=DEVICE)
        averaged = wp.zeros(N, dtype=float, device=DEVICE)

        # ---- 1. EAGER PASS: execute + record the ledger -------------------
        ledger: list[tuple] = []

        def logged_launch(kernel, dim, inputs, outputs):
            ledger.append((kernel, dim, list(inputs), list(outputs)))
            wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=DEVICE)

        logged_launch(add_fields, N, [a, b], [summed])
        logged_launch(scale_with_bias, N, [summed, bias], [scaled])
        logged_launch(finalize, N, [scaled], [averaged])
        wp.synchronize_device(DEVICE)

        eager_result = averaged.numpy().copy()  # legal: no capture open here
        reference = (a_np + b_np) * 2.0
        reference = (reference + bias_np) * 0.5
        eager_ok = bool(np.allclose(eager_result, reference))

        # ---- 2. RESET tracked inputs + CONTROLLED REPLAY into one .wrp ----
        a.assign(a_np)  # restore stored initial values (general-case reset)
        b.assign(b_np)

        fused_path = str(workdir / "fused_chain")
        with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as cap:
            for kernel, dim, inputs, outputs in ledger:
                wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=DEVICE)
        # Only tracked arrays are named; bias / summed / scaled are NOT params.
        wp.capture_save(
            cap.graph,
            fused_path,
            inputs={"a": a, "b": b},
            outputs={"averaged": averaged},
        )

        # Mutate the ORIGINAL bias after capture: a baked constant must ignore this.
        bias.assign(np.full(N, 999.0, dtype=np.float32))

        # ---- 3. VERIFY from a fresh load ---------------------------------
        a2 = wp.array(a_np, dtype=float, device=DEVICE)
        b2 = wp.array(b_np, dtype=float, device=DEVICE)
        loaded = wp.capture_load(fused_path, device=DEVICE)
        loaded.set_param("a", a2)
        loaded.set_param("b", b2)
        wp.capture_launch(loaded)
        wp.synchronize_device(DEVICE)
        out = wp.empty(N, dtype=float, device=DEVICE)
        loaded.get_param("averaged", out)
        loaded_result = out.numpy()

        fused_ok = bool(np.allclose(loaded_result, reference))
        params = list(getattr(loaded, "params", []) or [])
        # bias must NOT be a runtime param (it is baked)
        param_ok = ("a" in params and "b" in params and "averaged" in params
                    and "bias" not in params)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("=" * 72)
    print(f"eager chain matches reference                : {'PASS' if eager_ok else 'FAIL'}")
    print(f"    eager  = {eager_result.tolist()}")
    print(f"    ref    = {reference.tolist()}")
    print(f"fused .wrp (replay-from-log) matches reference: {'PASS' if fused_ok else 'FAIL'}")
    print(f"    loaded = {loaded_result.tolist()}")
    print(f"baked constant correct (bias not a param,")
    print(f"  and post-capture bias mutation ignored)    : {'PASS' if param_ok and fused_ok else 'FAIL'}")
    print(f"    loaded.params = {params}")
    print("=" * 72)


if __name__ == "__main__":
    main()
