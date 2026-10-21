# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate small ONNX bundles for the native integration tests."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def port(name, shape):
    return {"name": name, "dtype": "float32", "shape": shape, "type": "tensor"}


def save_model(root, name, inputs, outputs, nodes, initializers=(), external=False):
    def value(spec):
        shape = [None if dim == -1 else dim for dim in spec["shape"]]
        return helper.make_tensor_value_info(spec["name"], TensorProto.FLOAT, shape)

    graph = helper.make_graph(
        nodes,
        name,
        list(map(value, inputs)),
        list(map(value, outputs)),
        list(initializers),
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    directory = root / name
    directory.mkdir(exist_ok=True)
    # External data serialization appends; discard the previous fixture first.
    (directory / "weights.bin").unlink(missing_ok=True)
    path = directory / "model.onnx"
    onnx.save_model(
        model,
        path,
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    onnx.checker.check_model(path)
    return {
        "inputs": inputs,
        "outputs": outputs,
        "parameters": {"backend": "onnx", "model_path": f"{name}/model.onnx"},
    }


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    preprocess = save_model(
        root,
        "preprocess",
        [port("observation", [4])],
        [port("features", [4])],
        [helper.make_node("Add", ["observation", "one"], ["features"])],
        [numpy_helper.from_array(np.ones(4, dtype=np.float32), "one")],
    )
    policy = save_model(
        root,
        "policy",
        [port("features", [4]), port("bias", [4])],
        [port("action", [4]), port("echo", [4])],
        [
            helper.make_node("Mul", ["features", "scale"], ["scaled"]),
            helper.make_node("Add", ["scaled", "bias"], ["action"]),
            helper.make_node("Sub", ["features", "bias"], ["echo"]),
        ],
        [numpy_helper.from_array(np.array([2, 3, 4, 5], dtype=np.float32), "scale")],
        external=True,
    )
    bundle = {
        "models": {"policy": policy, "preprocess": preprocess},
        "pipeline": {
            "data_flow": {
                "preprocess/features": ["policy/features"],
                "policy/echo": [],
            },
            "feedback_flow": {},
            "inputs": {"preprocess": ["observation"], "policy": ["bias"]},
            "outputs": {"policy": ["action"]},
            "configs": {},
        },
    }

    def write(name, data):
        data["system information"] = {"leapp config version": "1.3"}
        (root / f"{name}.yaml").write_text(json.dumps(data, indent=2) + "\n")

    write("policy", bundle)
    defaults = copy.deepcopy(bundle)
    del defaults["pipeline"]["inputs"]["policy"]
    write("defaults", defaults)
    accumulator = save_model(
        root,
        "accumulator",
        [port("observation", [4]), port("state", [4])],
        [port("next", [4])],
        [helper.make_node("Add", ["observation", "state"], ["next"])],
    )
    save_model(
        root,
        "dynamic_accumulator",
        [port("observation", [-1]), port("state", [-1])],
        [port("next", [-1])],
        [helper.make_node("Add", ["observation", "state"], ["next"])],
    )
    write(
        "feedback",
        {
            "models": {"accumulator": accumulator},
            "pipeline": {
                "data_flow": {},
                "feedback_flow": {"accumulator/next": ["accumulator/state"]},
                "inputs": {"accumulator": ["observation"]},
                "outputs": {"accumulator": ["next"]},
                "configs": {},
            },
        },
    )
    save_model(root, "passthrough", [port("input", [-1])], [port("input", [-1])], [])
    reshape = save_model(
        root,
        "reshape",
        [port("input", [-1])],
        [port("output", [-1, 2])],
        [helper.make_node("Reshape", ["input", "shape"], ["output"])],
        [numpy_helper.from_array(np.array([-1, 2], dtype=np.int64), "shape")],
    )
    write(
        "reshape",
        {
            "models": {"reshape": reshape},
            "pipeline": {
                "data_flow": {},
                "feedback_flow": {},
                "inputs": {"reshape": ["input"]},
                "outputs": {"reshape": ["output"]},
                "configs": {},
            },
        },
    )
    for name, shape in [("scalar", []), ("dynamic", [-1]), ("empty", [0])]:
        model = save_model(
            root,
            name,
            [port("input", shape)],
            [port("output", shape)],
            [helper.make_node("Identity", ["input"], ["output"])],
        )
        write(
            name,
            {
                "models": {name: model},
                "pipeline": {
                    "data_flow": {},
                    "feedback_flow": {},
                    "inputs": {name: ["input"]},
                    "outputs": {name: ["output"]},
                    "configs": {},
                },
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    generate(parser.parse_args().output)
