# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Warp <-> PyTorch inference interop.

Pipeline shape:

    input torch tensor
        |-- Warp kernel A --|
        |-- Warp kernel B --|  (launched on independent streams)
                 |
          Warp merge kernel
                 |
          dense PyTorch forward
                 |
          final Warp postprocess kernel

The Warp stages operate on zero-copy wrappers around CUDA torch tensors via
``wp.from_torch``. This script measures eager Python dispatch overhead
and framework hand-off cost in addition to the actual kernel/model work.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
import warp as wp


DEVICE = "cuda:0"
FEATURE_DIM = 256
DENSE_HIDDEN_DIM = 512
DENSE_OUT_DIM = 128


@wp.kernel
def branch_wave_features(
    x: wp.array2d(dtype=float),
    out: wp.array2d(dtype=float),
):
    row, col = wp.tid()
    v = x[row, col]
    phase = float(row) * 0.0007 + float(col) * 0.013
    out[row, col] = wp.tanh(v * 1.31 + wp.sin(v + phase)) + 0.05 * wp.cos(v * 0.7)


@wp.kernel
def branch_stencil_features(
    x: wp.array2d(dtype=float),
    out: wp.array2d(dtype=float),
):
    row, col = wp.tid()
    center = x[row, col]
    left = center
    right = center

    if col > 0:
        left = x[row, col - 1]
    if col + 1 < FEATURE_DIM:
        right = x[row, col + 1]

    smoothed = center * 0.5 + (left + right) * 0.25
    out[row, col] = smoothed * smoothed + wp.sin(center * 0.25)


@wp.kernel
def merge_parallel_features(
    a: wp.array2d(dtype=float),
    b: wp.array2d(dtype=float),
    merged: wp.array2d(dtype=float),
):
    row, col = wp.tid()
    av = a[row, col]
    bv = b[row, col]
    gate = 1.0 / (1.0 + wp.exp(-(av * bv)))
    delta = av - bv
    merged[row, col] = gate * av + (1.0 - gate) * bv + 0.1 * delta * delta


@wp.kernel
def final_postprocess(
    dense_out: wp.array2d(dtype=float),
    final_out: wp.array2d(dtype=float),
):
    row, col = wp.tid()
    y = dense_out[row, col]
    compressed = wp.tanh(y) * 1.25
    energy = (y * y) / (1.0 + wp.abs(y))
    final_out[row, col] = compressed + 0.05 * energy


class DenseForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(FEATURE_DIM, DENSE_HIDDEN_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(DENSE_HIDDEN_DIM, DENSE_HIDDEN_DIM),
            torch.nn.GELU(),
            torch.nn.Linear(DENSE_HIDDEN_DIM, DENSE_OUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PipelineBuffers:
    input_torch: torch.Tensor
    branch_a_torch: torch.Tensor
    branch_a: wp.array
    branch_b_torch: torch.Tensor
    branch_b: wp.array
    merged_torch: torch.Tensor
    merged: wp.array
    final_torch: torch.Tensor
    final: wp.array


@dataclass
class PipelineStreams:
    branch_a: wp.Stream
    branch_b: wp.Stream


def build_buffers(batch_size: int) -> PipelineBuffers:
    input_torch = torch.randn(batch_size, FEATURE_DIM, device=DEVICE, dtype=torch.float32)
    branch_a_torch = torch.empty_like(input_torch)
    branch_b_torch = torch.empty_like(input_torch)
    merged_torch = torch.empty_like(input_torch)
    final_torch = torch.empty(batch_size, DENSE_OUT_DIM, device=DEVICE, dtype=torch.float32)

    return PipelineBuffers(
        input_torch=input_torch,
        branch_a_torch=branch_a_torch,
        branch_a=wp.from_torch(branch_a_torch, dtype=wp.float32),
        branch_b_torch=branch_b_torch,
        branch_b=wp.from_torch(branch_b_torch, dtype=wp.float32),
        merged_torch=merged_torch,
        merged=wp.from_torch(merged_torch, dtype=wp.float32),
        final_torch=final_torch,
        final=wp.from_torch(final_torch, dtype=wp.float32),
    )


def build_model() -> DenseForward:
    torch.manual_seed(1234)
    model = DenseForward().to(DEVICE).eval()
    return model


@torch.inference_mode()
def run_pipeline(
    model: DenseForward,
    buffers: PipelineBuffers,
    streams: PipelineStreams,
) -> torch.Tensor:
    batch_size = buffers.input_torch.shape[0]
    launch_shape = (batch_size, FEATURE_DIM)
    input_wp = wp.from_torch(buffers.input_torch, dtype=wp.float32)

    with wp.ScopedStream(streams.branch_a):
        wp.launch(
            branch_wave_features,
            dim=launch_shape,
            inputs=[input_wp],
            outputs=[buffers.branch_a],
            device=DEVICE,
        )

    with wp.ScopedStream(streams.branch_b):
        wp.launch(
            branch_stencil_features,
            dim=launch_shape,
            inputs=[input_wp],
            outputs=[buffers.branch_b],
            device=DEVICE,
        )

    wp.synchronize_device(DEVICE)

    wp.launch(
        merge_parallel_features,
        dim=launch_shape,
        inputs=[buffers.branch_a, buffers.branch_b],
        outputs=[buffers.merged],
        device=DEVICE,
    )
    wp.synchronize_device(DEVICE)

    dense_out = model(buffers.merged_torch)
    torch.cuda.synchronize()

    dense_wp = wp.from_torch(dense_out.contiguous(), dtype=wp.float32)
    wp.launch(
        final_postprocess,
        dim=(batch_size, DENSE_OUT_DIM),
        inputs=[dense_wp],
        outputs=[buffers.final],
        device=DEVICE,
    )
    wp.synchronize_device(DEVICE)

    return buffers.final_torch


@torch.inference_mode()
def run_pipeline_timed(
    model: DenseForward,
    buffers: PipelineBuffers,
    streams: PipelineStreams,
) -> dict[str, float]:
    batch_size = buffers.input_torch.shape[0]
    launch_shape = (batch_size, FEATURE_DIM)
    input_wp = wp.from_torch(buffers.input_torch, dtype=wp.float32)

    t0 = time.perf_counter()
    with wp.ScopedStream(streams.branch_a):
        wp.launch(
            branch_wave_features,
            dim=launch_shape,
            inputs=[input_wp],
            outputs=[buffers.branch_a],
            device=DEVICE,
        )
    with wp.ScopedStream(streams.branch_b):
        wp.launch(
            branch_stencil_features,
            dim=launch_shape,
            inputs=[input_wp],
            outputs=[buffers.branch_b],
            device=DEVICE,
        )
    wp.synchronize_device(DEVICE)
    t1 = time.perf_counter()

    wp.launch(
        merge_parallel_features,
        dim=launch_shape,
        inputs=[buffers.branch_a, buffers.branch_b],
        outputs=[buffers.merged],
        device=DEVICE,
    )
    wp.synchronize_device(DEVICE)
    t2 = time.perf_counter()

    dense_out = model(buffers.merged_torch)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    dense_wp = wp.from_torch(dense_out.contiguous(), dtype=wp.float32)
    wp.launch(
        final_postprocess,
        dim=(batch_size, DENSE_OUT_DIM),
        inputs=[dense_wp],
        outputs=[buffers.final],
        device=DEVICE,
    )
    wp.synchronize_device(DEVICE)
    t4 = time.perf_counter()

    return {
        "branch_warp_ms": (t1 - t0) * 1000.0,
        "merge_warp_ms": (t2 - t1) * 1000.0,
        "dense_torch_ms": (t3 - t2) * 1000.0,
        "final_warp_ms": (t4 - t3) * 1000.0,
        "total_ms": (t4 - t0) * 1000.0,
    }


def summarize(samples: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    keys = samples[0].keys()
    return {
        key: (
            statistics.mean(sample[key] for sample in samples),
            statistics.stdev(sample[key] for sample in samples) if len(samples) > 1 else 0.0,
        )
        for key in keys
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-capable PyTorch install.")

    wp.init()
    torch.manual_seed(2026)

    model = build_model()
    buffers = build_buffers(args.batch_size)
    streams = PipelineStreams(
        branch_a=wp.Stream(DEVICE),
        branch_b=wp.Stream(DEVICE),
    )

    for _ in range(args.warmup):
        run_pipeline(model, buffers, streams)

    samples = [run_pipeline_timed(model, buffers, streams) for _ in range(args.iterations)]
    summary = summarize(samples)

    print("Warp/PyTorch interop benchmark")
    print(f"batch_size={args.batch_size}, feature_dim={FEATURE_DIM}, dense_out_dim={DENSE_OUT_DIM}")
    for name, (mean_ms, stdev_ms) in summary.items():
        print(f"{name:>16}: {mean_ms:8.3f} ms +/- {stdev_ms:6.3f}")
    print(f"output_checksum: {buffers.final_torch.sum().item():.6f}")


if __name__ == "__main__":
    main()
