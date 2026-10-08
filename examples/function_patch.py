#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Capture existing OpenCV, NumPy, and torch code with one function patch.

Run from the repository root::

    python -m pip install opencv-python-headless
    python examples/function_patch.py

The stereo workflow calls cv2.cvtColor() inside nested image helpers. A single
patch captures both camera calls without changing those helpers. The exported
model uses torch operations and does not call OpenCV at runtime.
"""

import cv2
import numpy as np
import torch

import leapp
from leapp import InferenceManager, annotate


# LEAPP adapter: implement only the OpenCV operation this workflow needs.
def traced_cvt_color(src, code, dst=None, dstCn=0):
    """Trace BGR-to-RGB conversion for three-channel HWC images."""
    if code != cv2.COLOR_BGR2RGB or dst is not None or dstCn not in (0, 3):
        raise ValueError("This adapter supports only BGR2RGB without a destination buffer")
    if src.ndim != 3 or src.shape[-1] != 3:
        raise ValueError("This adapter expects an HWC image with three channels")
    return torch.from_numpy(src).flip((-1,)).numpy()


# Existing vision workflow: no LEAPP annotations or tracing branches here.
def prepare_image(bgr: np.ndarray) -> np.ndarray:
    """Convert a camera frame to RGB and arrange channels for a torch model."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.transpose(rgb, (2, 0, 1))


def image_features(bgr: np.ndarray) -> torch.Tensor:
    """Extract a small spatial color descriptor for one camera."""
    image = torch.from_numpy(prepare_image(bgr)).float() / 255.0
    pooled = torch.nn.functional.adaptive_avg_pool2d(image.unsqueeze(0), (2, 2))
    return pooled.flatten()


def stereo_features(left_bgr, right_bgr):
    """Existing workflow: OpenCV preprocessing followed by torch fusion."""
    return torch.cat([image_features(left_bgr), image_features(right_bgr)])


def main():
    rng = np.random.default_rng(42)
    sample = {
        name: rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
        for name in ("left_bgr", "right_bgr")
    }
    ordinary_result = stereo_features(*sample.values())
    original_cvt_color = cv2.cvtColor
    leapp.start(
        name="function_patch_example",
        patching=[leapp.FunctionPatch(cv2, "cvtColor", traced_cvt_color)],
    )
    try:
        traced = annotate.input_tensors("camera_preprocess", sample)
        features = stereo_features(*traced)
        annotate.output_tensors(
            "camera_preprocess", {"stereo_features": features}, export_with="jit",
        )
        # Ordinary arrays still take the original OpenCV path during tracing.
        torch.testing.assert_close(stereo_features(*sample.values()), ordinary_result)
    finally:
        leapp.stop()

    assert cv2.cvtColor is original_cvt_color
    leapp.compile_graph(visualize=False, validate=True)

    manager = InferenceManager(
        "function_patch_example/function_patch_example.yaml",
    )
    inputs = manager.get_mock_input()
    fresh = {
        name: rng.integers(0, 256, size=image.shape, dtype=image.dtype)
        for name, image in sample.items()
    }
    for name, image in fresh.items():
        key = f"camera_preprocess/{name}"
        inputs[key] = inputs[key].new_tensor(image)
    actual = manager.run_policy(inputs)["camera_preprocess/stereo_features"]
    expected = stereo_features(*fresh.values())
    torch.testing.assert_close(actual, expected.to(actual.device))
    print("Patch restored:", cv2.cvtColor is original_cvt_color)
    print("Replayed stereo features:", actual)


if __name__ == "__main__":
    main()
