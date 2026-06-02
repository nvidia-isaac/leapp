"""Run the embedded-bundle WrpRunner ONNX model through ONNX Runtime.

To prove the model is self-contained, the model is copied into a fresh temp dir
(along with its `<model>.onnx.data` sidecar if external data was used) with no
`.wrp` / `_modules` present, and executed from there. The custom op reconstructs
each APIC bundle from the bytes carried in the model's initializers.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

# Import torch before onnxruntime so CUDA/cuDNN wheel libraries are loaded into
# the process. Without this, ORT's CUDA provider may fail to dlopen dependencies
# such as libcudnn.so.9 in this conda environment.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import onnxruntime as ort

from generate_wrp import (
    HEIGHT,
    SMALL_HEIGHT,
    SMALL_WIDTH,
    WIDTH,
    expected_result,
    expected_small_result,
)


def _make_session(model_path: Path, custom_op_library: Path) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(str(custom_op_library.resolve()))
    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def run(model_path: Path, custom_op_library: Path, portable_check: bool) -> None:
    run_model_path = model_path
    temp_dir = None

    if portable_check:
        # Copy ONLY the model (+ its external-data sidecar) into an isolated dir
        # with no sibling .wrp / _modules. ORT resolves <model>.onnx.data
        # relative to the copied model, so the bundle still loads.
        temp_dir = Path(tempfile.mkdtemp(prefix="wrp_portable_"))
        run_model_path = temp_dir / model_path.name
        shutil.copy2(model_path, run_model_path)

        data_sidecar = model_path.with_name(model_path.name + ".data")
        copied = [model_path.name]
        if data_sidecar.exists():
            shutil.copy2(data_sidecar, temp_dir / data_sidecar.name)
            copied.append(data_sidecar.name)
        print(f"portable check: running from {temp_dir} with {copied} (no .wrp present)")

    try:
        session = _make_session(run_model_path, custom_op_library)

        a = np.arange(HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)
        b = np.full((HEIGHT, WIDTH), 10.0, dtype=np.float32)
        x = np.arange(SMALL_HEIGHT * SMALL_WIDTH, dtype=np.float32).reshape(
            SMALL_HEIGHT, SMALL_WIDTH
        )
        y = np.full((SMALL_HEIGHT, SMALL_WIDTH), 2.0, dtype=np.float32)

        averaged, squared = session.run(None, {"a": a, "b": b, "x": x, "y": y})
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    expected_averaged = expected_result(a, b)
    expected_squared = expected_small_result(x, y)

    print("ONNX Runtime kernel_chain output:")
    print(averaged)
    print("ONNX Runtime subtract_square output:")
    print(squared)
    np.testing.assert_allclose(averaged, expected_averaged, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(squared, expected_squared, rtol=1e-6, atol=1e-6)
    print("matched expected outputs from both embedded .wrp bundles")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--model", type=Path, default=root / "artifacts" / "wrp_runner_embedded.onnx"
    )
    parser.add_argument(
        "--custom-op-library",
        type=Path,
        default=root / "build" / "libwrp_onnx_custom_op.so",
    )
    parser.add_argument(
        "--no-portable-check",
        action="store_true",
        help="Run the model in place instead of copying it to an isolated dir.",
    )
    args = parser.parse_args()
    run(args.model, args.custom_op_library, portable_check=not args.no_portable_check)


if __name__ == "__main__":
    main()
