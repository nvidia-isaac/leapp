# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build TensorRT engines and bundles for Triton tests on the target GPU."""

import argparse
import json
from pathlib import Path
import subprocess

from generate_models import generate as generate_onnx


def generate(root, trtexec):
    root = root.resolve()
    source = root / "onnx"
    generate_onnx(source)

    engines = {}
    for name in ["preprocess", "policy", "accumulator", "dynamic"]:
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        engine = directory / ("policy.engine" if name == "policy" else "model.plan")
        command = [
            trtexec,
            f"--onnx={source / name / 'model.onnx'}",
            f"--saveEngine={engine}",
            "--skipInference",
            "--noTF32",
            "--builderOptimizationLevel=0",
            "--profilingVerbosity=none",
        ]
        if name == "dynamic":
            command += [
                "--minShapes=input:1",
                "--optShapes=input:2",
                "--maxShapes=input:4",
            ]
        subprocess.run(command, check=True)
        engines[name] = engine.relative_to(root).as_posix()

    def write(name, original, converted):
        bundle = json.loads((source / f"{original}.yaml").read_text())
        for node, model in bundle["models"].items():
            parameters = model["parameters"]
            if node in converted:
                parameters.update(backend="trt", model_path=engines[node])
            else:
                parameters["model_path"] = "onnx/" + parameters["model_path"]
        (root / f"{name}.yaml").write_text(json.dumps(bundle, indent=2) + "\n")

    write("policy", "policy", {"preprocess", "policy"})
    write("mixed", "policy", {"policy"})
    write("feedback", "feedback", {"accumulator"})
    write("dynamic", "dynamic", {"dynamic"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trtexec", default="trtexec")
    args = parser.parse_args()
    generate(args.output, args.trtexec)
