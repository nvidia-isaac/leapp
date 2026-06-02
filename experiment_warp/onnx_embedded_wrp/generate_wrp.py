"""Generate Warp APIC graphs for the embedded-bundle WrpRunner prototype.

This is the same set of tiny kernels as the sibling generic_onnx_op prototype.
The difference is purely downstream: make_onnx.py embeds the resulting .wrp
bundles directly inside the ONNX model instead of referencing them by path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp


HEIGHT = 4
WIDTH = 5
SMALL_HEIGHT = 2
SMALL_WIDTH = 3
DEVICE = "cuda:0"


@wp.kernel
def add_fields(
    a: wp.array2d(dtype=float),
    b: wp.array2d(dtype=float),
    summed: wp.array2d(dtype=float),
):
    row, col = wp.tid()
    summed[row, col] = a[row, col] + b[row, col]


@wp.kernel
def scale_and_bias(
    summed: wp.array2d(dtype=float),
    scale: float,
    bias: float,
    scaled: wp.array2d(dtype=float),
):
    row, col = wp.tid()
    scaled[row, col] = summed[row, col] * scale + bias


@wp.kernel
def neighbor_average(
    scaled: wp.array2d(dtype=float),
    averaged: wp.array2d(dtype=float),
):
    row, col = wp.tid()

    center = scaled[row, col]
    left = center
    right = center
    up = center
    down = center

    if col > 0:
        left = scaled[row, col - 1]
    if col + 1 < WIDTH:
        right = scaled[row, col + 1]
    if row > 0:
        up = scaled[row - 1, col]
    if row + 1 < HEIGHT:
        down = scaled[row + 1, col]

    averaged[row, col] = (center + left + right + up + down) * 0.2


def expected_result(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scaled = (a + b) * 2.0 + 1.0
    averaged = np.empty_like(scaled)
    for row in range(HEIGHT):
        for col in range(WIDTH):
            center = scaled[row, col]
            left = scaled[row, col - 1] if col > 0 else center
            right = scaled[row, col + 1] if col + 1 < WIDTH else center
            up = scaled[row - 1, col] if row > 0 else center
            down = scaled[row + 1, col] if row + 1 < HEIGHT else center
            averaged[row, col] = (center + left + right + up + down) * 0.2
    return averaged


def run_kernel_chain(a, b, summed, scaled, averaged) -> None:
    shape = (HEIGHT, WIDTH)
    wp.launch(add_fields, dim=shape, inputs=[a, b], outputs=[summed], device=DEVICE)
    wp.launch(scale_and_bias, dim=shape, inputs=[summed, 2.0, 1.0], outputs=[scaled], device=DEVICE)
    wp.launch(neighbor_average, dim=shape, inputs=[scaled], outputs=[averaged], device=DEVICE)


@wp.kernel
def subtract_and_square(
    x: wp.array2d(dtype=float),
    y: wp.array2d(dtype=float),
    squared: wp.array2d(dtype=float),
):
    row, col = wp.tid()
    diff = x[row, col] - y[row, col]
    squared[row, col] = diff * diff


def expected_small_result(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.square(x - y)


def run_small_graph(x, y, squared) -> None:
    wp.launch(
        subtract_and_square,
        dim=(SMALL_HEIGHT, SMALL_WIDTH),
        inputs=[x, y],
        outputs=[squared],
        device=DEVICE,
    )


def generate_kernel_chain(output_base: Path) -> None:
    a_np = np.arange(HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)
    b_np = np.full((HEIGHT, WIDTH), 10.0, dtype=np.float32)

    a = wp.array(a_np, dtype=float, device=DEVICE)
    b = wp.array(b_np, dtype=float, device=DEVICE)
    summed = wp.zeros((HEIGHT, WIDTH), dtype=float, device=DEVICE)
    scaled = wp.zeros((HEIGHT, WIDTH), dtype=float, device=DEVICE)
    averaged = wp.zeros((HEIGHT, WIDTH), dtype=float, device=DEVICE)

    with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as capture:
        run_kernel_chain(a, b, summed, scaled, averaged)

    wp.capture_save(
        capture.graph,
        str(output_base),
        inputs={"a": a, "b": b},
        outputs={"averaged": averaged},
    )

    print(f"wrote {output_base.with_suffix('.wrp')}")
    print("expected kernel_chain output for default inputs:")
    print(expected_result(a_np, b_np))


def generate_small_graph(output_base: Path) -> None:
    x_np = np.arange(SMALL_HEIGHT * SMALL_WIDTH, dtype=np.float32).reshape(
        SMALL_HEIGHT, SMALL_WIDTH
    )
    y_np = np.full((SMALL_HEIGHT, SMALL_WIDTH), 2.0, dtype=np.float32)

    x = wp.array(x_np, dtype=float, device=DEVICE)
    y = wp.array(y_np, dtype=float, device=DEVICE)
    squared = wp.zeros((SMALL_HEIGHT, SMALL_WIDTH), dtype=float, device=DEVICE)

    with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as capture:
        run_small_graph(x, y, squared)

    wp.capture_save(
        capture.graph,
        str(output_base),
        inputs={"x": x, "y": y},
        outputs={"squared": squared},
    )

    print(f"wrote {output_base.with_suffix('.wrp')}")
    print("expected subtract_square output for default inputs:")
    print(expected_small_result(x_np, y_np))


def generate_wrps(output_dir: Path) -> None:
    wp.init()
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_kernel_chain(output_dir / "kernel_chain")
    generate_small_graph(output_dir / "subtract_square")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Directory for generated .wrp files and module directories.",
    )
    args = parser.parse_args()
    generate_wrps(args.output_dir)


if __name__ == "__main__":
    main()
