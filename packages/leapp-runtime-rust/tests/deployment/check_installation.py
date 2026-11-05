# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a minimal installation using pre-generated fixtures, without exporters.

Run in an isolated environment containing only the selected model runtime and
its backend plugins. See the runtime contributor guide for the test matrix.
"""

import argparse
import os
from pathlib import Path

import lrt
import numpy as np


def fixture(backend, device="cpu"):
    variable, directory, filename = {
        "onnx": ("LEAPP_TRITON_FIXTURES", "triton-fixtures", "feedback"),
        "torchscript": (
            "LEAPP_TORCHSCRIPT_FIXTURES",
            "torchscript-fixtures",
            "mutating",
        ),
        "nativert": (
            "LEAPP_PT2_FIXTURES",
            "pt2-fixtures",
            "weighted_cuda" if device.startswith("cuda") else "weighted_cpu",
        ),
        "tensorrt": ("LEAPP_TENSORRT_FIXTURES", "tensorrt-fixtures", "feedback"),
    }[backend]
    root = Path(os.environ.get(variable, f"target/{directory}"))
    return lrt.Bundle.open(root / f"{filename}.yaml")


def execute(backend, engine, device):
    bundle = fixture(backend, device)
    name, output, expected = {
        "onnx": ("accumulator/observation", "accumulator/next", [1, 2, 3, 4]),
        "tensorrt": ("accumulator/observation", "accumulator/next", [1, 2, 3, 4]),
        "torchscript": ("model/values", "model/result", [9, 12, 15, 18]),
        "nativert": ("model/values", "model/output_0", [12, 26, 42, 60]),
    }[backend]
    values = np.arange(1, 5, dtype=np.float32)
    inputs = {name: values}
    with lrt.Session.open(
        bundle, lrt.SessionOptions(engine=engine, device=device)
    ) as session:
        peer = session.new_session()
        retained = np.from_dlpack(session.run(inputs)[output])
        np.testing.assert_array_equal(retained, expected)
    with peer:
        np.testing.assert_array_equal(
            np.from_dlpack(peer.run(inputs)[output]), expected
        )
    np.testing.assert_array_equal(retained, expected)
    np.testing.assert_array_equal(values, [1, 2, 3, 4])
    print(f"PASS {engine}/{backend}/{device}: inference, peer, retained output")
    if backend == "onnx":
        root = Path(os.environ.get("LEAPP_TRITON_FIXTURES", "target/triton-fixtures"))
        convolution = lrt.Bundle.open(root / "convolution.yaml")
        values = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        with lrt.Session.open(
            convolution, lrt.SessionOptions(engine=engine, device=device)
        ) as session:
            result = session.run({"convolution/input": values})
            np.testing.assert_array_equal(
                np.from_dlpack(result["convolution/output"]),
                [[[[10, 14, 18], [26, 30, 34], [42, 46, 50]]]],
            )
        print(f"PASS {engine}/{backend}/{device}: convolution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["none", "onnx", "torchscript", "nativert", "tensorrt"],
        required=True,
    )
    parser.add_argument("--engine", choices=["native", "triton"], required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.backend == "tensorrt" and not args.cuda:
        parser.error("TensorRT execution requires --cuda")

    # Even absent backends must permit bundle inspection. A CPU session would
    # reject TensorRT placement before reaching its loader, so use CUDA there.
    for backend in ["onnx", "torchscript", "nativert", "tensorrt"]:
        bundle = fixture(backend)
        assert bundle.model_names(), backend
        if backend == args.backend or (backend == "tensorrt" and not args.cuda):
            continue
        device = "cuda:0" if backend == "tensorrt" else "cpu"
        try:
            with lrt.Session.open(
                bundle, lrt.SessionOptions(engine=args.engine, device=device)
            ):
                raise AssertionError(f"unexpectedly loaded missing {backend} backend")
        except lrt.RuntimeError as error:
            assert error.phase == "load", str(error)
            assert any(node in str(error) for node in bundle.model_names()), str(error)
        print(f"PASS {args.engine}/{backend}: inspection and missing-backend error")

    if args.backend != "none":
        devices = [] if args.backend == "tensorrt" else ["cpu"]
        if args.cuda:
            devices.append("cuda:0")
        for device in devices:
            execute(args.backend, args.engine, device)


if __name__ == "__main__":
    main()
