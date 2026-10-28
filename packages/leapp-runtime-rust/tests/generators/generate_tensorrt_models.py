# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build TensorRT engines and bundles for execution tests on the target GPU."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from onnx import TensorProto, helper
import tensorrt as trt

from generate_models import generate as generate_onnx, port, save_model


def generate(root, trtexec):
    root = root.resolve()
    source = root / "onnx"
    generate_onnx(source)

    def single(name, inputs, outputs, nodes, initializers=()):
        model = save_model(source, name, inputs, outputs, nodes, initializers)
        bundle = {
            "models": {name: model},
            "pipeline": {
                "inputs": {name: [p["name"] for p in inputs]},
                "outputs": {name: [p["name"] for p in outputs]},
                "data_flow": {},
                "feedback_flow": {},
            },
            "system information": {"leapp config version": "1.3"},
        }
        (source / f"{name}.yaml").write_text(json.dumps(bundle))

    single(
        "concat",
        [port("left", [-1]), port("right", [-1])],
        [port("output", [-1])],
        [helper.make_node("Concat", ["left", "right"], ["output"], axis=0)],
    )
    single(
        "nonzero",
        [port("data", [4])],
        [port("indices", [1, -1], "int64")],
        [helper.make_node("NonZero", ["data"], ["indices"])],
    )
    single(
        "shape",
        [port("data", [-1, 2])],
        [port("output", [2], "int32")],
        [
            helper.make_node("Shape", ["data"], ["dimensions"]),
            helper.make_node("Cast", ["dimensions"], ["output"], to=TensorProto.INT32),
        ],
    )
    engines = {}
    for name in [
        "preprocess",
        "policy",
        "accumulator",
        "empty",
        "scalar",
    ]:
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
        subprocess.run(command, check=True)
        engines[name] = engine.relative_to(root).as_posix()

    def build(name, profiles):
        logger = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(logger) as builder, builder.create_network(0) as network:
            with trt.OnnxParser(network, logger) as parser:
                if not parser.parse_from_file(str(source / name / "model.onnx")):
                    raise RuntimeError(
                        [str(parser.get_error(i)) for i in range(parser.num_errors)]
                    )
                if name == "shape":
                    output = network.get_output(0)
                    network.unmark_output(output)
                    if not network.mark_output_for_shapes(output):
                        raise RuntimeError(
                            "TensorRT could not mark the host shape output"
                        )
                config = builder.create_builder_config()
                config.clear_flag(trt.BuilderFlag.TF32)
                config.builder_optimization_level = 0
                for bounds in profiles:
                    profile = builder.create_optimization_profile()
                    for index in range(network.num_inputs):
                        tensor = network.get_input(index)
                        sizes = bounds.get(tensor.name, [list(tensor.shape)] * 3)
                        if tensor.is_shape_tensor:
                            profile.set_shape_input(tensor.name, *sizes)
                        else:
                            profile.set_shape(tensor.name, *sizes)
                    assert config.add_optimization_profile(profile) >= 0
                serialized = builder.build_serialized_network(network, config)
                if serialized is None:
                    raise RuntimeError(f"TensorRT could not build {name}")
                directory = root / name
                directory.mkdir(exist_ok=True)
                (directory / "model.plan").write_bytes(bytes(serialized))
                engines[name] = f"{name}/model.plan"
                if name == "shape":
                    with trt.Runtime(logger) as runtime:
                        engine = runtime.deserialize_cuda_engine(serialized)
                        assert (
                            engine.get_tensor_location("output")
                            == trt.TensorLocation.HOST
                        )
                        assert engine.is_shape_inference_io("output")

    build("dynamic", [{"input": [[1], [2], [4]]}, {"input": [[5], [6], [8]]}])
    build(
        "concat",
        [
            {"left": [[1], [1], [2]], "right": [[1], [2], [3]]},
            {"left": [[1], [2], [3]], "right": [[4], [5], [6]]},
        ],
    )
    build(
        "slice",
        [
            {"start": [[0], [0], [0]], "end": [[2], [2], [2]]},
            {"start": [[1], [1], [2]], "end": [[3], [3], [4]]},
        ],
    )
    build("nonzero", [{}])
    build("shape", [{"data": [[1, 2], [2, 2], [4, 2]]}])

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
    for name in [
        "dynamic",
        "concat",
        "slice",
        "nonzero",
        "shape",
        "empty",
        "scalar",
    ]:
        write(name, name, {name})
    # ONNX RandomUniform with a static shape can be folded at build time.
    # A runtime alpha input keeps a real IFillLayer and its context-owned RNG.
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, builder.create_network(0) as network:
        low = network.add_input("low", trt.float32, (1,))
        scalar = network.add_shuffle(low)
        scalar.reshape_dims = ()
        fill = network.add_fill((16,), trt.FillOperation.RANDOM_UNIFORM)
        fill.set_input(1, scalar.get_output(0))
        high = network.add_constant((), np.asarray(1.0, dtype=np.float32))
        fill.set_input(2, high.get_output(0))
        fill.get_output(0).name = "output"
        network.mark_output(fill.get_output(0))
        config = builder.create_builder_config()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT could not build the stateful RNG fixture")
        directory = root / "random"
        directory.mkdir(exist_ok=True)
        (directory / "model.plan").write_bytes(bytes(serialized))
    random = {
        "models": {
            "random": {
                "inputs": [port("low", [1])],
                "outputs": [port("output", [16])],
                "parameters": {"backend": "trt", "model_path": "random/model.plan"},
            }
        },
        "pipeline": {
            "inputs": {"random": ["low"]},
            "outputs": {"random": ["output"]},
            "data_flow": {},
            "feedback_flow": {},
        },
        "system information": {"leapp config version": "1.3"},
    }
    (root / "random.yaml").write_text(json.dumps(random, indent=2) + "\n")

    fixed = json.loads((root / "nonzero.yaml").read_text())
    fixed["models"]["nonzero"]["outputs"][0]["shape"] = [1, 2]
    (root / "nonzero_fixed.yaml").write_text(json.dumps(fixed, indent=2) + "\n")
    shape_fixed = json.loads((root / "shape.yaml").read_text())
    shape_fixed["models"]["shape"]["inputs"][0]["shape"] = [2, 2]
    (root / "shape_fixed.yaml").write_text(json.dumps(shape_fixed, indent=2) + "\n")
    pinned = json.loads((root / "dynamic.yaml").read_text())
    for direction in ["inputs", "outputs"]:
        pinned["models"]["dynamic"][direction][0]["shape"] = [8]
    (root / "dynamic_fixed.yaml").write_text(json.dumps(pinned, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trtexec", default="trtexec")
    args = parser.parse_args()
    generate(args.output, args.trtexec)
