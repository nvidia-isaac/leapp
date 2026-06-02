# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate artifacts for the Warp ONNX vs straight C++ benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import warp as wp

from benchmark_pipeline import (
    DENSE_OUT_DIM,
    DEVICE,
    FEATURE_DIM,
    DenseForward,
    branch_stencil_features,
    branch_wave_features,
    final_postprocess,
    merge_parallel_features,
)


def _capture_branch_wave(output_base: Path, batch_size: int, x_np: np.ndarray) -> None:
    x = wp.array(x_np, dtype=float, device=DEVICE)
    dummy = wp.zeros((batch_size, FEATURE_DIM), dtype=float, device=DEVICE)
    out = wp.zeros((batch_size, FEATURE_DIM), dtype=float, device=DEVICE)
    with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as capture:
        wp.launch(
            branch_wave_features,
            dim=(batch_size, FEATURE_DIM),
            inputs=[x],
            outputs=[out],
            device=DEVICE,
        )
    wp.capture_save(capture.graph, str(output_base), inputs={"input": x, "dummy": dummy}, outputs={"output": out})


def _capture_branch_stencil(output_base: Path, batch_size: int, x_np: np.ndarray) -> None:
    x = wp.array(x_np, dtype=float, device=DEVICE)
    dummy = wp.zeros((batch_size, FEATURE_DIM), dtype=float, device=DEVICE)
    out = wp.zeros((batch_size, FEATURE_DIM), dtype=float, device=DEVICE)
    with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as capture:
        wp.launch(
            branch_stencil_features,
            dim=(batch_size, FEATURE_DIM),
            inputs=[x],
            outputs=[out],
            device=DEVICE,
        )
    wp.capture_save(capture.graph, str(output_base), inputs={"input": x, "dummy": dummy}, outputs={"output": out})


def _capture_merge(output_base: Path, batch_size: int) -> None:
    a = wp.zeros((batch_size, FEATURE_DIM), dtype=float, device=DEVICE)
    b = wp.zeros((batch_size, FEATURE_DIM), dtype=float, device=DEVICE)
    out = wp.zeros((batch_size, FEATURE_DIM), dtype=float, device=DEVICE)
    with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as capture:
        wp.launch(
            merge_parallel_features,
            dim=(batch_size, FEATURE_DIM),
            inputs=[a, b],
            outputs=[out],
            device=DEVICE,
        )
    wp.capture_save(capture.graph, str(output_base), inputs={"branch_a": a, "branch_b": b}, outputs={"merged": out})


def _capture_final(output_base: Path, batch_size: int) -> None:
    dense_out = wp.zeros((batch_size, DENSE_OUT_DIM), dtype=float, device=DEVICE)
    dummy = wp.zeros((batch_size, DENSE_OUT_DIM), dtype=float, device=DEVICE)
    final = wp.zeros((batch_size, DENSE_OUT_DIM), dtype=float, device=DEVICE)
    with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as capture:
        wp.launch(
            final_postprocess,
            dim=(batch_size, DENSE_OUT_DIM),
            inputs=[dense_out],
            outputs=[final],
            device=DEVICE,
        )
    wp.capture_save(
        capture.graph,
        str(output_base),
        inputs={"dense_out": dense_out, "dummy": dummy},
        outputs={"final": final},
    )


def _export_dense(output_path: Path, batch_size: int) -> None:
    torch.manual_seed(1234)
    model = DenseForward().to(DEVICE).eval()
    example = torch.zeros(batch_size, FEATURE_DIM, device=DEVICE, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["merged"],
        output_names=["dense_out"],
        opset_version=18,
        dynamic_axes=None,
    )


def generate(output_dir: Path, batch_size: int) -> None:
    wp.init()
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)
    x_np = rng.normal(size=(batch_size, FEATURE_DIM)).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "input.npy", x_np)
    x_np.tofile(output_dir / "input.bin")
    _capture_branch_wave(output_dir / "branch_wave_features", batch_size, x_np)
    _capture_branch_stencil(output_dir / "branch_stencil_features", batch_size, x_np)
    _capture_merge(output_dir / "merge_parallel_features", batch_size)
    _capture_final(output_dir / "final_postprocess", batch_size)
    _export_dense(output_dir / "dense.onnx", batch_size)

    print(f"wrote benchmark artifacts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    generate(args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
