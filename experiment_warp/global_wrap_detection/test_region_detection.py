"""Region-accuracy test for the exact interleaving the user described:

    launch ; launch ; launch
    <unrelated pytorch cuda op on unrelated arrays>
    launch ; launch
    <print using related arrays>
    launch ; launch                       (7 launches total)

Question: can this be ONE region (one fused .wrp)? We answer physically by
holding ONE capture open across an interleaved op and seeing what survives.
Each warp launch does x += 1 on a shared tracked array, so a valid fused .wrp
turns 0 -> 7.

Each factor is isolated in its OWN subprocess (a capture error dirties the CUDA
context, so scenarios must not share a process). Factors:
  baseline        : 7 launches only
  torch_only      : one unrelated torch cuda op (in-place) mid-sequence
  print_contents  : print(x)         -> wp.array.__str__ calls x.numpy() (D2H)
  print_numpy     : print(x.numpy()) -> explicit D2H
  print_meta      : access x.shape   -> metadata only, no device read

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate exp_env
    python experiment_warp/global_wrap_detection/test_region_detection.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import warp as wp


DEVICE = "cuda:0"
N = 8


@wp.kernel
def add_one(x: wp.array(dtype=float)):
    i = wp.tid()
    x[i] = x[i] + 1.0


def run_one(scenario: str) -> tuple[bool, str]:
    wp.init()
    do_torch = scenario == "torch_only"
    print_mode = {
        "print_contents": "contents",
        "print_numpy": "numpy",
        "print_meta": "meta",
    }.get(scenario)

    do_torch_side = scenario == "torch_side_stream"
    t = None
    side_stream = None
    if do_torch or do_torch_side:
        import torch  # noqa: PLC0415
        t = torch.ones(256, 256, device=DEVICE)
        if do_torch_side:
            side_stream = torch.cuda.Stream()

    workdir = Path(tempfile.mkdtemp(prefix="region_one_"))
    x = wp.zeros(N, dtype=float, device=DEVICE)
    graph = None
    try:
        wp.capture_begin(device=DEVICE, force_module_load=True, apic=True)
        wp.launch(add_one, dim=N, inputs=[x], device=DEVICE)   # 1
        wp.launch(add_one, dim=N, inputs=[x], device=DEVICE)   # 2
        wp.launch(add_one, dim=N, inputs=[x], device=DEVICE)   # 3
        if do_torch and t is not None:
            t.add_(1.0)
        if do_torch_side and t is not None:
            import torch  # noqa: PLC0415
            with torch.cuda.stream(side_stream):
                t.add_(1.0)
        wp.launch(add_one, dim=N, inputs=[x], device=DEVICE)   # 4
        wp.launch(add_one, dim=N, inputs=[x], device=DEVICE)   # 5
        if print_mode == "contents":
            print("    [in-capture] x =", x)
        elif print_mode == "numpy":
            print("    [in-capture] x =", x.numpy())
        elif print_mode == "meta":
            _ = x.shape
        wp.launch(add_one, dim=N, inputs=[x], device=DEVICE)   # 6
        wp.launch(add_one, dim=N, inputs=[x], device=DEVICE)   # 7
        graph = wp.capture_end(device=DEVICE)
    except Exception as exc:  # noqa: BLE001
        return False, f"capture raised: {type(exc).__name__}: {str(exc).splitlines()[0]}"

    try:
        path = str(workdir / "region")
        wp.capture_save(graph, path, inputs={"x": x}, outputs={"x": x})
        result = wp.zeros(N, dtype=float, device=DEVICE)
        loaded = wp.capture_load(path, device=DEVICE)
        loaded.set_param("x", result)
        wp.capture_launch(loaded)
        wp.synchronize_device(DEVICE)
        out = wp.empty(N, dtype=float, device=DEVICE)
        loaded.get_param("x", out)
        vals = out.numpy()
    except Exception as exc:  # noqa: BLE001
        return False, f"save/replay raised: {type(exc).__name__}: {str(exc).splitlines()[0]}"

    ok = bool(np.allclose(vals, 7.0))
    return ok, f"fused replay -> {vals.tolist()}"


SCENARIOS = [
    ("baseline: 7 launches only",                 "baseline"),
    ("unrelated torch cuda op (default stream)",  "torch_only"),
    ("unrelated torch cuda op (side stream)",     "torch_side_stream"),
    ("print(x)  [CONTENTS -> numpy() D2H]",        "print_contents"),
    ("print(x.numpy())  [explicit D2H]",          "print_numpy"),
    ("x.shape  [METADATA only, no read]",         "print_meta"),
]


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        ok, detail = run_one(sys.argv[2])
        print(f"RESULT|{'OK' if ok else 'BREAK'}|{detail}")
        return

    print("=" * 80)
    print("Can all 7 launches stay in ONE capture region with the op interleaved?")
    print("(each scenario in a fresh subprocess)")
    print("-" * 80)
    for label, key in SCENARIOS:
        proc = subprocess.run(
            [sys.executable, __file__, "--one", key],
            capture_output=True, text=True,
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT|")), None)
        if line is None:
            verdict, detail = "ERR", (proc.stderr.strip().splitlines() or ["<no output>"])[-1]
        else:
            _, v, detail = line.split("|", 2)
            verdict = "1-REGION OK" if v == "OK" else "BREAKS"
        print(f"  {verdict:<12} | {label}")
        print(f"               {detail}")
    print("=" * 80)


if __name__ == "__main__":
    main()
