# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a minimal installation using pre-generated fixtures, without exporters.

Run in an isolated environment containing only the selected model runtime and
its backend plugins. Native CUDA checks also require CuPy for device inputs.
See the runtime contributor guide for the test matrix.
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


def execute(backend, engine, device, inspect):
    xp = np
    if engine == "native" and device.startswith("cuda"):
        import cupy as xp

    bundle = fixture(backend, device)
    name, output, expected = {
        "onnx": ("accumulator/observation", "accumulator/next", [1, 2, 3, 4]),
        "tensorrt": ("accumulator/observation", "accumulator/next", [1, 2, 3, 4]),
        "torchscript": ("model/values", "model/result", [9, 12, 15, 18]),
        "nativert": ("model/values", "model/output_0", [12, 26, 42, 60]),
    }[backend]
    values = xp.asarray(np.arange(1, 5, dtype=np.float32))
    inputs = {name: values}
    with lrt.Session.open(
        bundle, lrt.SessionOptions(engine=engine, device=device)
    ) as session:
        peer = session.new_session()
        retained = np.from_dlpack(session.run(inputs)[output])
        np.testing.assert_array_equal(retained, expected)
        if backend == "onnx" and engine == "native":
            mappings = Path("/proc/self/maps").read_text()
            assert "libonnxruntime" in mappings
            if device.startswith("cuda"):
                assert "libonnxruntime_providers_cuda.so" in mappings
        inspect()
    with peer:
        np.testing.assert_array_equal(
            np.from_dlpack(peer.run(inputs)[output]), expected
        )
    np.testing.assert_array_equal(retained, expected)
    np.testing.assert_array_equal(
        xp.asnumpy(values) if xp is not np else values, [1, 2, 3, 4]
    )
    print(f"PASS {engine}/{backend}/{device}: inference, peer, retained output")
    if backend == "onnx":
        root = Path(os.environ.get("LEAPP_TRITON_FIXTURES", "target/triton-fixtures"))
        convolution = lrt.Bundle.open(root / "convolution.yaml")
        values = xp.asarray(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
        with lrt.Session.open(
            convolution, lrt.SessionOptions(engine=engine, device=device)
        ) as session:
            result = session.run({"convolution/input": values})
            np.testing.assert_array_equal(
                np.from_dlpack(result["convolution/output"]),
                [[[[10, 14, 18], [26, 30, 34], [42, 46, 50]]]],
            )
            inspect()
        print(f"PASS {engine}/{backend}/{device}: convolution")


def execute_mixed(engine, inspect):
    xp = np
    if engine == "native":
        import cupy as xp

    root = Path(os.environ.get("LEAPP_TENSORRT_FIXTURES", "target/tensorrt-fixtures"))
    with lrt.Session.open(
        lrt.Bundle.open(root / "mixed.yaml"),
        lrt.SessionOptions(engine=engine, device="cuda:0"),
    ) as session:
        result = session.run(
            {
                "preprocess/observation": xp.arange(1, 5, dtype=xp.float32),
                "policy/bias": xp.arange(10, 50, 10, dtype=xp.float32),
            }
        )
        np.testing.assert_array_equal(
            np.from_dlpack(result["policy/action"]), [14, 29, 46, 65]
        )
        inspect()
    print(f"PASS {engine}/onnx+tensorrt/cuda: mixed graph")


def execute_warp(backend, engine, inspect):
    root = Path(os.environ["LEAPP_WARP_FIXTURES"])
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    gains = np.arange(1, 5, dtype=np.int32)
    for device in ["cpu", "cuda:0"]:
        fixture = (
            "warp_onnx"
            if backend == "onnx"
            else "warp_pt2_" + ("cpu" if device == "cpu" else "cuda")
        )
        with lrt.Session.open(
            lrt.Bundle.open(root / f"{fixture}.yaml"),
            lrt.SessionOptions(engine=engine, device=device),
        ) as session:
            result = session.run({"model/x": values, "model/gain": gains})
            np.testing.assert_array_equal(
                np.from_dlpack(result["model/result"]), values * gains[:, None] + 4
            )
            inspect()
        print(f"PASS {engine}/warp-{backend}/{device}")


def check_unloaded(without_triton, without_cuda):
    libraries = []
    if without_triton:
        libraries.append("libtritonserver.so")
    if without_cuda:
        libraries.extend(["libcuda.so", "libcudart.so"])
    if libraries:
        mappings = Path("/proc/self/maps").read_text()
        for library in libraries:
            assert library not in mappings, f"unexpectedly loaded {library}"


def missing_dependency(options, library, run=False):
    try:
        with lrt.Session.open(fixture("onnx"), options) as session:
            if run:
                session.run(
                    {"accumulator/observation": np.arange(1, 5, dtype=np.float32)}
                )
            raise AssertionError(f"unexpectedly succeeded without {library}")
    except lrt.RuntimeError as error:
        assert error.code == lrt.ErrorCode.UNSUPPORTED, str(error)
        assert library in str(error), str(error)
    print(f"PASS missing {library}: requested operation returns a dependency error")


def main(inspect=lambda: None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["none", "onnx", "torchscript", "nativert", "tensorrt"],
        required=True,
        nargs="+",
    )
    parser.add_argument("--engine", choices=["native", "triton"], required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--without-triton", action="store_true")
    parser.add_argument("--without-cuda", action="store_true")
    parser.add_argument(
        "--installed-only",
        action="store_true",
        help="Only require fixtures for selected backends; skip missing-backend checks.",
    )
    args = parser.parse_args(argv)
    if "tensorrt" in args.backend:
        if args.engine != "triton":
            parser.error("precompiled TensorRT engines require --engine triton")
        if not args.cuda:
            parser.error("TensorRT execution requires --cuda")

    if args.without_triton and args.engine != "native":
        parser.error("--without-triton requires --engine native")
    if args.without_cuda and (
        args.cuda or args.engine != "native" or args.backend != ["onnx"]
    ):
        parser.error("--without-cuda requires native CPU ONNX")

    check_unloaded(args.without_triton, args.without_cuda)
    if args.without_triton:
        missing_dependency(lrt.SessionOptions(engine="triton"), "libtritonserver.so")
    if args.without_cuda:
        missing_dependency(
            lrt.SessionOptions(engine="native", device="cuda:0"), "libcudart.so"
        )
        missing_dependency(
            lrt.SessionOptions(engine="native", output_device="pinned"),
            "libcudart.so",
            run=True,
        )

    # Even absent backends must permit bundle inspection. A CPU session would
    # reject TensorRT placement before reaching its loader, so use CUDA there.
    for backend in ["onnx", "torchscript", "nativert", "tensorrt"]:
        if args.installed_only and backend not in args.backend:
            continue
        bundle = fixture(backend)
        assert bundle.model_names(), backend
        if backend in args.backend or (
            backend == "tensorrt" and (not args.cuda or args.engine == "native")
        ):
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

    for backend in args.backend:
        if backend == "none":
            continue
        devices = [] if backend == "tensorrt" else ["cpu"]
        if args.cuda:
            devices.append("cuda:0")
        for device in devices:
            execute(backend, args.engine, device, inspect)

    if {"onnx", "tensorrt"} <= set(args.backend):
        execute_mixed(args.engine, inspect)
    check_unloaded(args.without_triton, args.without_cuda)


if __name__ == "__main__":
    main()
