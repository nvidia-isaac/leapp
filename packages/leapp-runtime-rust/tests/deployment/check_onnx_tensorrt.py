# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check TensorRT compilation, caching, profiles, and engine-specific options."""

import argparse
from pathlib import Path

import lrt
import numpy as np


def check(fixtures, cache, engine):
    bundle = lrt.Bundle.open(fixtures / "named_ports_dynamic.yaml")
    left = np.arange(4, dtype=np.float32)
    right = left * 2
    inputs = {"arithmetic/left": left, "arithmetic/right": right}
    expected = {"arithmetic/sum": left + right, "arithmetic/difference": left - right}
    cases = (
        [(None, True)]
        if engine == "triton"
        else [("Identity", True), ("Add,Sub", False)]
    )
    for excluded, expect_engine in cases:
        directory = cache / (excluded or "default")
        directory.mkdir(parents=True, exist_ok=True)
        parameters = {
            "precision_mode": "FP16",
            "trt_engine_cache_enable": "1",
            "trt_engine_cache_path": str(directory),
            "trt_builder_optimization_level": "0",
            **{
                f"trt_profile_{kind}_shapes": f"left:{size},right:{size}"
                for kind, size in [("min", 1), ("opt", 4), ("max", 8)]
            },
        }
        if excluded is not None:
            parameters["trt_op_types_to_exclude"] = excluded
        options = lrt.SessionOptions(
            engine=engine, device="cuda:0", onnx_tensorrt={"arithmetic": parameters}
        )
        snapshot = None
        for _ in range(2):
            with lrt.Session.open(bundle, options) as session:
                result = session.run(inputs)
                for port, values in expected.items():
                    np.testing.assert_allclose(np.from_dlpack(result[port]), values)
            engines = {p.name: p.stat().st_mtime_ns for p in directory.glob("*.engine")}
            assert bool(engines) == expect_engine, f"Unexpected engines: {engines}"
            if snapshot is not None:
                assert engines == snapshot, "Cache was rebuilt on warm open"
            snapshot = engines
        print(f"PASS {engine}: profiles, cache reuse, operator exclusion {excluded}")
    invalid_options = [({"precision_mode": "invalid"}, "precision")]
    if engine == "native":
        invalid_options.extend(
            [
                ({"unknown_option": "1"}, "Unknown provider option"),
                ({"unknown_option": ""}, "key/value cannot be empty"),
            ]
        )
    else:
        invalid_options.extend(
            [
                ({"unknown_option": "1"}, "unknown parameter"),
                ({"unknown_option": ""}, "unknown parameter"),
                ({"trt_op_types_to_exclude": "Add,Sub"}, "unknown parameter"),
            ]
        )
    for parameters, expected_error in invalid_options:
        try:
            with lrt.Session.open(
                bundle,
                lrt.SessionOptions(
                    engine=engine,
                    device="cuda:0",
                    onnx_tensorrt={"arithmetic": parameters},
                ),
            ):
                raise AssertionError(f"Invalid parameters accepted: {parameters}")
        except lrt.RuntimeError as error:
            assert expected_error in str(error), str(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--engine", choices=["native", "triton"], required=True)
    args = parser.parse_args()
    check(args.fixtures, args.cache, args.engine)
