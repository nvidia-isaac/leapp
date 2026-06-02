"""Run the WrpRunner ONNX model through ONNX Runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from generate_wrp import (
    HEIGHT,
    SMALL_HEIGHT,
    SMALL_WIDTH,
    WIDTH,
    expected_result,
    expected_small_result,
)


def run(model_path: Path, custom_op_library: Path) -> None:
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(str(custom_op_library.resolve()))

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    a = np.arange(HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)
    b = np.full((HEIGHT, WIDTH), 10.0, dtype=np.float32)
    x = np.arange(SMALL_HEIGHT * SMALL_WIDTH, dtype=np.float32).reshape(
        SMALL_HEIGHT, SMALL_WIDTH
    )
    y = np.full((SMALL_HEIGHT, SMALL_WIDTH), 2.0, dtype=np.float32)

    averaged, squared = session.run(None, {"a": a, "b": b, "x": x, "y": y})
    expected_averaged = expected_result(a, b)
    expected_squared = expected_small_result(x, y)

    print("ONNX Runtime kernel_chain output:")
    print(averaged)
    print("ONNX Runtime subtract_square output:")
    print(squared)
    np.testing.assert_allclose(averaged, expected_averaged, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(squared, expected_squared, rtol=1e-6, atol=1e-6)
    print("matched expected outputs from both .wrp files")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--model", type=Path, default=root / "artifacts" / "wrp_runner.onnx")
    parser.add_argument(
        "--custom-op-library",
        type=Path,
        default=root / "build" / "libwrp_onnx_custom_op.so",
    )
    args = parser.parse_args()
    run(args.model, args.custom_op_library)


if __name__ == "__main__":
    main()
