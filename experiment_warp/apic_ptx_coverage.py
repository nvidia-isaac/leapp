"""Probe Warp APIC `.wrp` replay coverage for cubin vs PTX modules.

Run from a Warp/APIC environment, for example:

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate exp_env
    python experiment_warp/apic_ptx_coverage.py

The script captures a set of common Warp operations twice: once with
`wp.config.cuda_output = "cubin"` and once with `"ptx"`. Each case is saved to
disk, loaded back from the `.wrp`, launched, and checked against a NumPy
reference. For PTX cases, successful replay means the saved PTX-backed APIC
bundle was loaded and executed by the CUDA driver.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


DEVICE = "cuda:0"
N = 16
HEIGHT = 4
WIDTH = 5


def _import_warp(cuda_output: str, ptx_target_arch: int | None):
    import warp as wp  # noqa: PLC0415

    wp.config.cuda_output = cuda_output
    if ptx_target_arch is not None:
        wp.config.ptx_target_arch = ptx_target_arch
    wp.init()
    return wp


try:
    import warp as wp

    @wp.kernel
    def scalar_math_kernel(
        a: wp.array(dtype=wp.float32),
        b: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        out[i] = a[i] * b[i] + wp.sin(a[i]) - wp.cos(b[i]) + wp.sqrt(a[i] + 2.0)

    @wp.kernel
    def vector_math_kernel(
        a: wp.array(dtype=wp.vec3),
        b: wp.array(dtype=wp.vec3),
        out: wp.array(dtype=wp.vec3),
        lengths: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        av = a[i]
        bv = b[i]
        crossed = wp.cross(av, bv)
        scaled = wp.normalize(av + wp.vec3(0.1, 0.2, 0.3)) * wp.dot(av, bv)
        out[i] = crossed + scaled
        lengths[i] = wp.length(out[i])

    @wp.kernel
    def stencil_2d_kernel(
        src: wp.array2d(dtype=wp.float32),
        out: wp.array2d(dtype=wp.float32),
    ):
        row, col = wp.tid()
        center = src[row, col]
        left = center
        right = center
        up = center
        down = center
        if col > 0:
            left = src[row, col - 1]
        if col + 1 < WIDTH:
            right = src[row, col + 1]
        if row > 0:
            up = src[row - 1, col]
        if row + 1 < HEIGHT:
            down = src[row + 1, col]
        out[row, col] = (center + left + right + up + down) * 0.2

    @wp.kernel
    def control_atomic_kernel(
        src: wp.array(dtype=wp.float32),
        per_item: wp.array(dtype=wp.float32),
        bins: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        acc = float(0.0)
        for j in range(4):
            acc = acc + src[i] * float(j + 1)
        if i % 2 == 0:
            acc = -acc
        per_item[i] = acc
        wp.atomic_add(bins, i % 4, acc)

    @wp.kernel
    def chain_add_kernel(
        a: wp.array(dtype=wp.float32),
        b: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        out[i] = a[i] + b[i]

    @wp.kernel
    def chain_scale_kernel(
        src: wp.array(dtype=wp.float32),
        scale: wp.float32,
        bias: wp.float32,
        out: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        out[i] = src[i] * scale + bias

    @wp.kernel
    def chain_square_kernel(
        src: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        out[i] = src[i] * src[i]

    @wp.kernel
    def tiled_add_kernel(
        src: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.float32),
    ):
        i, tile = wp.tid()
        if tile == 0:
            out[i] = src[i] + 2.0

except ModuleNotFoundError:
    wp = None


@dataclass
class CaseResult:
    name: str
    cuda_output: str
    ok: bool
    stage: str
    detail: str
    wrp: str
    modules: list[str]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


def _clean_base(base: Path) -> None:
    base.with_suffix(".wrp").unlink(missing_ok=True)
    shutil.rmtree(base.parent / f"{base.name}_modules", ignore_errors=True)


def _module_files(base: Path) -> list[str]:
    module_dir = base.parent / f"{base.name}_modules"
    if not module_dir.exists():
        return []
    return sorted(str(path.relative_to(base.parent)) for path in module_dir.iterdir() if path.is_file())


def _check_module_format(files: list[str], cuda_output: str) -> tuple[bool, str]:
    if cuda_output == "ptx":
        if any(name.endswith(".ptx") for name in files):
            return True, "found .ptx module"
        return False, "expected at least one .ptx module"
    if cuda_output == "cubin":
        if any(name.endswith(".cubin") for name in files):
            return True, "found .cubin module"
        return False, "expected at least one .cubin module"
    return True, "module format not checked"


def _allclose(actual: np.ndarray, expected: np.ndarray, atol: float = 1.0e-4) -> tuple[bool, str]:
    ok = bool(np.allclose(actual, expected, rtol=1.0e-4, atol=atol))
    max_err = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    return ok, f"max_err={max_err:.3g}"


def _capture_save_load_run(
    *,
    wp_mod,
    base: Path,
    run_capture: Callable[[], None],
    inputs: dict[str, object],
    outputs: dict[str, object],
) -> object:
    _clean_base(base)
    with wp_mod.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as capture:
        run_capture()
    wp_mod.capture_save(capture.graph, str(base), inputs=inputs, outputs=outputs)
    graph = wp_mod.capture_load(str(base), device=DEVICE)
    for name, value in inputs.items():
        graph.set_param(name, value)
    wp_mod.capture_launch(graph)
    wp_mod.synchronize_device(DEVICE)
    return graph


def case_scalar_math(wp_mod, base: Path) -> tuple[bool, str]:
    a_np = np.linspace(0.1, 2.0, N, dtype=np.float32)
    b_np = np.linspace(1.5, 0.2, N, dtype=np.float32)
    expected = a_np * b_np + np.sin(a_np) - np.cos(b_np) + np.sqrt(a_np + 2.0)
    a = wp_mod.array(a_np, dtype=wp_mod.float32, device=DEVICE)
    b = wp_mod.array(b_np, dtype=wp_mod.float32, device=DEVICE)
    out = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)
    graph = _capture_save_load_run(
        wp_mod=wp_mod,
        base=base,
        run_capture=lambda: wp_mod.launch(scalar_math_kernel, dim=N, inputs=[a, b], outputs=[out], device=DEVICE),
        inputs={"a": a, "b": b},
        outputs={"out": out},
    )
    replayed = wp_mod.empty(N, dtype=wp_mod.float32, device=DEVICE)
    graph.get_param("out", replayed)
    return _allclose(replayed.numpy(), expected)


def case_vector_math(wp_mod, base: Path) -> tuple[bool, str]:
    a_np = np.stack([np.linspace(0.2, 1.7, N), np.linspace(1.0, 2.0, N), np.linspace(2.0, 3.0, N)], axis=1).astype(np.float32)
    b_np = np.stack([np.linspace(1.1, 0.3, N), np.linspace(0.5, 1.5, N), np.linspace(2.5, 0.7, N)], axis=1).astype(np.float32)
    shifted = a_np + np.array([0.1, 0.2, 0.3], dtype=np.float32)
    shifted_norm = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)
    expected_vec = np.cross(a_np, b_np) + shifted_norm * np.sum(a_np * b_np, axis=1, keepdims=True)
    expected_len = np.linalg.norm(expected_vec, axis=1)
    a = wp_mod.array(a_np, dtype=wp_mod.vec3, device=DEVICE)
    b = wp_mod.array(b_np, dtype=wp_mod.vec3, device=DEVICE)
    out = wp_mod.zeros(N, dtype=wp_mod.vec3, device=DEVICE)
    lengths = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)
    graph = _capture_save_load_run(
        wp_mod=wp_mod,
        base=base,
        run_capture=lambda: wp_mod.launch(vector_math_kernel, dim=N, inputs=[a, b], outputs=[out, lengths], device=DEVICE),
        inputs={"a": a, "b": b},
        outputs={"out": out, "lengths": lengths},
    )
    replayed_vec = wp_mod.empty(N, dtype=wp_mod.vec3, device=DEVICE)
    replayed_len = wp_mod.empty(N, dtype=wp_mod.float32, device=DEVICE)
    graph.get_param("out", replayed_vec)
    graph.get_param("lengths", replayed_len)
    ok_vec, detail_vec = _allclose(replayed_vec.numpy(), expected_vec)
    ok_len, detail_len = _allclose(replayed_len.numpy(), expected_len)
    return ok_vec and ok_len, f"vec_{detail_vec}; len_{detail_len}"


def case_stencil_2d(wp_mod, base: Path) -> tuple[bool, str]:
    src_np = np.arange(HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)
    expected = np.empty_like(src_np)
    for row in range(HEIGHT):
        for col in range(WIDTH):
            center = src_np[row, col]
            left = src_np[row, col - 1] if col > 0 else center
            right = src_np[row, col + 1] if col + 1 < WIDTH else center
            up = src_np[row - 1, col] if row > 0 else center
            down = src_np[row + 1, col] if row + 1 < HEIGHT else center
            expected[row, col] = (center + left + right + up + down) * 0.2
    src = wp_mod.array(src_np, dtype=wp_mod.float32, device=DEVICE)
    out = wp_mod.zeros((HEIGHT, WIDTH), dtype=wp_mod.float32, device=DEVICE)
    graph = _capture_save_load_run(
        wp_mod=wp_mod,
        base=base,
        run_capture=lambda: wp_mod.launch(stencil_2d_kernel, dim=(HEIGHT, WIDTH), inputs=[src], outputs=[out], device=DEVICE),
        inputs={"src": src},
        outputs={"out": out},
    )
    replayed = wp_mod.empty((HEIGHT, WIDTH), dtype=wp_mod.float32, device=DEVICE)
    graph.get_param("out", replayed)
    return _allclose(replayed.numpy(), expected)


def case_control_atomic(wp_mod, base: Path) -> tuple[bool, str]:
    src_np = np.linspace(0.25, 2.0, N, dtype=np.float32)
    expected_items = src_np * 10.0
    expected_items[::2] *= -1.0
    expected_bins = np.zeros(4, dtype=np.float32)
    for i, value in enumerate(expected_items):
        expected_bins[i % 4] += value
    src = wp_mod.array(src_np, dtype=wp_mod.float32, device=DEVICE)
    per_item = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)
    bins = wp_mod.zeros(4, dtype=wp_mod.float32, device=DEVICE)
    graph = _capture_save_load_run(
        wp_mod=wp_mod,
        base=base,
        run_capture=lambda: wp_mod.launch(control_atomic_kernel, dim=N, inputs=[src], outputs=[per_item, bins], device=DEVICE),
        inputs={"src": src},
        outputs={"per_item": per_item, "bins": bins},
    )
    replayed_items = wp_mod.empty(N, dtype=wp_mod.float32, device=DEVICE)
    replayed_bins = wp_mod.empty(4, dtype=wp_mod.float32, device=DEVICE)
    graph.get_param("per_item", replayed_items)
    graph.get_param("bins", replayed_bins)
    ok_items, detail_items = _allclose(replayed_items.numpy(), expected_items)
    ok_bins, detail_bins = _allclose(replayed_bins.numpy(), expected_bins)
    return ok_items and ok_bins, f"items_{detail_items}; bins_{detail_bins}"


def case_multi_launch_chain(wp_mod, base: Path) -> tuple[bool, str]:
    a_np = np.linspace(0.0, 1.5, N, dtype=np.float32)
    b_np = np.linspace(2.0, 4.0, N, dtype=np.float32)
    expected = np.square((a_np + b_np) * 1.75 - 0.5)
    a = wp_mod.array(a_np, dtype=wp_mod.float32, device=DEVICE)
    b = wp_mod.array(b_np, dtype=wp_mod.float32, device=DEVICE)
    tmp0 = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)
    tmp1 = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)
    out = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)

    def run() -> None:
        wp_mod.launch(chain_add_kernel, dim=N, inputs=[a, b], outputs=[tmp0], device=DEVICE)
        wp_mod.launch(chain_scale_kernel, dim=N, inputs=[tmp0, 1.75, -0.5], outputs=[tmp1], device=DEVICE)
        wp_mod.launch(chain_square_kernel, dim=N, inputs=[tmp1], outputs=[out], device=DEVICE)

    graph = _capture_save_load_run(
        wp_mod=wp_mod,
        base=base,
        run_capture=run,
        inputs={"a": a, "b": b},
        outputs={"out": out},
    )
    replayed = wp_mod.empty(N, dtype=wp_mod.float32, device=DEVICE)
    graph.get_param("out", replayed)
    return _allclose(replayed.numpy(), expected)


def case_launch_tiled(wp_mod, base: Path) -> tuple[bool, str]:
    src_np = np.linspace(-2.0, 2.0, N, dtype=np.float32)
    expected = src_np + 2.0
    src = wp_mod.array(src_np, dtype=wp_mod.float32, device=DEVICE)
    out = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)
    graph = _capture_save_load_run(
        wp_mod=wp_mod,
        base=base,
        run_capture=lambda: wp_mod.launch_tiled(tiled_add_kernel, dim=(N,), inputs=[src], outputs=[out], block_dim=4, device=DEVICE),
        inputs={"src": src},
        outputs={"out": out},
    )
    replayed = wp_mod.empty(N, dtype=wp_mod.float32, device=DEVICE)
    graph.get_param("out", replayed)
    return _allclose(replayed.numpy(), expected)


def case_memory_ops(wp_mod, base: Path) -> tuple[bool, str]:
    src_np = np.linspace(1.0, 3.0, N, dtype=np.float32)
    assign_np = np.linspace(-3.0, -1.0, N, dtype=np.float32)
    src = wp_mod.array(src_np, dtype=wp_mod.float32, device=DEVICE)
    assign_src = wp_mod.array(assign_np, dtype=wp_mod.float32, device=DEVICE)
    copied = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)
    filled = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)
    assigned = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)

    def run() -> None:
        wp_mod.copy(copied, src)
        filled.fill_(3.5)
        assigned.assign(assign_src)

    graph = _capture_save_load_run(
        wp_mod=wp_mod,
        base=base,
        run_capture=run,
        inputs={"src": src, "assign_src": assign_src},
        outputs={"copied": copied, "filled": filled, "assigned": assigned},
    )
    replayed_copied = wp_mod.empty(N, dtype=wp_mod.float32, device=DEVICE)
    replayed_filled = wp_mod.empty(N, dtype=wp_mod.float32, device=DEVICE)
    replayed_assigned = wp_mod.empty(N, dtype=wp_mod.float32, device=DEVICE)
    graph.get_param("copied", replayed_copied)
    graph.get_param("filled", replayed_filled)
    graph.get_param("assigned", replayed_assigned)
    ok_copy, detail_copy = _allclose(replayed_copied.numpy(), src_np)
    ok_fill, detail_fill = _allclose(replayed_filled.numpy(), np.full(N, 3.5, dtype=np.float32))
    ok_assign, detail_assign = _allclose(replayed_assigned.numpy(), assign_np)
    return ok_copy and ok_fill and ok_assign, f"copy_{detail_copy}; fill_{detail_fill}; assign_{detail_assign}"


def case_utility_array_sum(wp_mod, base: Path) -> tuple[bool, str]:
    src_np = np.linspace(1.0, 2.0, N, dtype=np.float32)
    expected = np.array([np.sum(src_np)], dtype=np.float32)
    src = wp_mod.array(src_np, dtype=wp_mod.float32, device=DEVICE)
    out = wp_mod.zeros(1, dtype=wp_mod.float32, device=DEVICE)
    graph = _capture_save_load_run(
        wp_mod=wp_mod,
        base=base,
        run_capture=lambda: wp_mod.utils.array_sum(src, out=out),
        inputs={"src": src},
        outputs={"out": out},
    )
    replayed = wp_mod.empty(1, dtype=wp_mod.float32, device=DEVICE)
    graph.get_param("out", replayed)
    return _allclose(replayed.numpy(), expected)


def case_utility_array_scan(wp_mod, base: Path) -> tuple[bool, str]:
    src_np = np.arange(1, N + 1, dtype=np.float32)
    expected = np.cumsum(src_np)
    src = wp_mod.array(src_np, dtype=wp_mod.float32, device=DEVICE)
    out = wp_mod.zeros(N, dtype=wp_mod.float32, device=DEVICE)
    graph = _capture_save_load_run(
        wp_mod=wp_mod,
        base=base,
        run_capture=lambda: wp_mod.utils.array_scan(src, out),
        inputs={"src": src},
        outputs={"out": out},
    )
    replayed = wp_mod.empty(N, dtype=wp_mod.float32, device=DEVICE)
    graph.get_param("out", replayed)
    return _allclose(replayed.numpy(), expected)


CASES: dict[str, Callable[[object, Path], tuple[bool, str]]] = {
    "scalar_math": case_scalar_math,
    "vector_math": case_vector_math,
    "stencil_2d": case_stencil_2d,
    "control_atomic": case_control_atomic,
    "multi_launch_chain": case_multi_launch_chain,
    "launch_tiled": case_launch_tiled,
    "memory_ops": case_memory_ops,
    "utility_array_sum": case_utility_array_sum,
    "utility_array_scan": case_utility_array_scan,
}


def run_case(case_name: str, cuda_output: str, output_dir: Path, ptx_target_arch: int | None) -> CaseResult:
    if wp is None:
        return CaseResult(case_name, cuda_output, False, "import", "warp is not installed", "", [])
    try:
        wp_mod = _import_warp(cuda_output, ptx_target_arch)
        if not wp_mod.is_cuda_available():
            return CaseResult(case_name, cuda_output, False, "cuda", "CUDA is not available", "", [])
        base = output_dir / cuda_output / case_name / case_name
        base.parent.mkdir(parents=True, exist_ok=True)
        ok, detail = CASES[case_name](wp_mod, base)
        files = _module_files(base)
        format_ok, format_detail = _check_module_format(files, cuda_output)
        final_ok = ok and format_ok
        final_detail = f"{detail}; {format_detail}"
        return CaseResult(case_name, cuda_output, final_ok, "replay", final_detail, str(base.with_suffix(".wrp")), files)
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            case_name,
            cuda_output,
            False,
            "exception",
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            "",
            [],
        )


def child_main(args: argparse.Namespace) -> int:
    result = run_case(args.case, args.cuda_output, args.output_dir, args.ptx_target_arch)
    print("RESULT|" + result.to_json())
    if args.verbose and not result.ok:
        traceback.print_exc()
    return 0 if result.ok else 1


def parent_main(args: argparse.Namespace) -> int:
    formats = ["cubin", "ptx"] if args.cuda_output == "both" else [args.cuda_output]
    case_names = args.cases or list(CASES)
    results: list[CaseResult] = []
    for cuda_output in formats:
        for case_name in case_names:
            cmd = [
                sys.executable,
                __file__,
                "--child",
                "--cuda-output",
                cuda_output,
                "--case",
                case_name,
                "--output-dir",
                str(args.output_dir),
            ]
            if args.ptx_target_arch is not None:
                cmd.extend(["--ptx-target-arch", str(args.ptx_target_arch)])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT|")), None)
            if line is None:
                detail = (proc.stderr.strip().splitlines() or proc.stdout.strip().splitlines() or ["no child output"])[-1]
                result = CaseResult(case_name, cuda_output, False, "child", detail, "", [])
            else:
                result = CaseResult(**json.loads(line.split("|", 1)[1]))
            results.append(result)
            status = "OK" if result.ok else "FAIL"
            print(f"{status:<4} {cuda_output:<5} {case_name:<20} {result.detail}")
            if result.modules:
                print(f"     modules: {', '.join(result.modules)}")

    print("\nSummary by case:")
    by_case: dict[str, dict[str, CaseResult]] = {}
    for result in results:
        by_case.setdefault(result.name, {})[result.cuda_output] = result
    for case_name in case_names:
        cubin = by_case.get(case_name, {}).get("cubin")
        ptx = by_case.get(case_name, {}).get("ptx")
        if cubin and ptx:
            relation = "same" if cubin.ok == ptx.ok else "differs"
            print(f"  {case_name:<20} cubin={cubin.ok!s:<5} ptx={ptx.ok!s:<5} {relation}")
        else:
            only = cubin or ptx
            if only:
                print(f"  {case_name:<20} {only.cuda_output}={only.ok}")
    return 0 if all(result.ok for result in results) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=sorted(CASES), help="Single case to run in child mode.")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASES),
        help="Subset of cases to run in parent mode.",
    )
    parser.add_argument(
        "--cuda-output",
        choices=["both", "cubin", "ptx"],
        default="both",
        help="CUDA module output format to request via wp.config.cuda_output.",
    )
    parser.add_argument(
        "--ptx-target-arch",
        type=int,
        default=None,
        help="Optional wp.config.ptx_target_arch value, e.g. 87 for Jetson Orin.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "apic_ptx_coverage_artifacts",
        help="Directory for generated .wrp files and module directories.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print traceback in child mode on failure.")
    args = parser.parse_args()
    if args.child and not args.case:
        parser.error("--child requires --case")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(child_main(parsed) if parsed.child else parent_main(parsed))
