# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export small TorchScript bundles and direct PyTorch reference outputs."""

import argparse
import copy
import json
from pathlib import Path

import torch
from generate_models import port, save_model
from onnx import helper


class Weighted(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor([2.0, 3.0, 4.0, 5.0]))

    def forward(self, values, bias):
        return values * self.weight + bias, values.reshape(2, 2).t()


class Mutating(torch.nn.Module):
    def forward(self, values):
        values.add_(2)
        return values * 3


class Producer(torch.nn.Module):
    def forward(self, values):
        return values + 0.0


class Observer(torch.nn.Module):
    def forward(self, original, dependency):
        return original + dependency * 0.0


class MutatingAccumulator(torch.nn.Module):
    def forward(self, observation, state):
        state.add_(observation)
        if bool(observation[0] < 0):
            raise RuntimeError("failure after mutation")
        return state


class Reshape(torch.nn.Module):
    def forward(self, values):
        return values.reshape(-1, 2)


class Forked(torch.nn.Module):
    def read(self, values):
        return values + 1

    def forward(self, values):
        future = torch.jit.fork(self.read, values)
        return torch.jit.wait(future)


class Integer(torch.nn.Module):
    def forward(self, values):
        return values + 1


def bundle(model, node="model"):
    return {
        "system information": {"leapp config version": "1.3"},
        "models": {node: model},
        "pipeline": {
            "data_flow": {},
            "feedback_flow": {},
            "inputs": {node: [p["name"] for p in model["inputs"]]},
            "outputs": {node: [p["name"] for p in model["outputs"]]},
            "configs": {},
        },
    }


def generate(root):
    root.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        (root / f"{name}.yaml").write_text(json.dumps(value, indent=2) + "\n")

    def export(name, module, inputs, outputs, examples, traced=False):
        module.eval()
        model = (
            torch.jit.trace(module, examples) if traced else torch.jit.script(module)
        )
        torch.jit.freeze(model).save(str(root / f"{name}.pt"))
        return {
            "inputs": inputs,
            "outputs": outputs,
            "parameters": {
                "backend": "jit-trace" if traced else "jit-script",
                "model_path": f"{name}.pt",
            },
        }

    values = torch.arange(1.0, 5.0)
    bias = torch.arange(1.0, 5.0) * 10
    # Names deliberately differ from forward arguments and ordinal-looking names.
    inputs = [port("z_values", [4]), port("INPUT__9", [4])]
    outputs = [port("z_result", [4]), port("OUTPUT__7", [2, 2])]
    for traced in [False, True]:
        name = "traced" if traced else "scripted"
        weighted = export(name, Weighted(), inputs, outputs, (values, bias), traced)
        write(name, bundle(weighted))

    expected = [t.flatten().tolist() for t in Weighted()(values, bias)]
    (root / "expected.json").write_text(json.dumps(expected) + "\n")
    for name, module, dtype, input_shape, output_shape in [
        ("mutating", Mutating(), "float32", [4], [4]),
        ("dynamic", Reshape(), "float32", [-1], [-1, 2]),
        ("integer", Integer(), "int64", [4], [4]),
        ("forked", Forked(), "float32", [4], [4]),
    ]:
        source = dict(port("values", input_shape), dtype=dtype)
        target = dict(port("result", output_shape), dtype=dtype)
        write(name, bundle(export(name, module, [source], [target], (values,))))

    preprocess = save_model(
        root,
        "preprocess",
        [port("values", [4])],
        [port("features", [4])],
        [helper.make_node("Identity", ["values"], ["features"])],
    )
    mixed = bundle(export("weighted", Weighted(), inputs, outputs, (values, bias)))
    mixed["models"]["preprocess"] = preprocess
    mixed["pipeline"]["inputs"] = {"model": ["INPUT__9"], "preprocess": ["values"]}
    mixed["pipeline"]["data_flow"] = {"preprocess/features": ["model/z_values"]}
    write("mixed", mixed)

    mutating_accumulator = export(
        "mutating_accumulator",
        MutatingAccumulator(),
        [port("observation", [4]), port("state", [4])],
        [port("next", [4])],
        (values, values),
    )
    mutating_feedback = bundle(mutating_accumulator, "accumulator")
    mutating_feedback["pipeline"]["inputs"] = {"accumulator": ["observation"]}
    mutating_feedback["pipeline"]["feedback_flow"] = {
        "accumulator/next": ["accumulator/state"]
    }
    write("mutating_feedback", mutating_feedback)

    producer = export(
        "producer", Producer(), [port("values", [4])], [port("result", [4])], (values,)
    )
    observer = export(
        "observer",
        Observer(),
        [port("original", [4]), port("dependency", [4])],
        [port("result", [4])],
        (values, values),
    )
    consumer = export(
        "mutating", Mutating(), [port("values", [4])], [port("result", [4])], (values,)
    )
    fanout = bundle(producer, "producer")
    fanout["models"].update(consumer=consumer, observer=observer)
    fanout["pipeline"]["data_flow"] = {
        "producer/result": ["consumer/values", "observer/original"],
        "consumer/result": ["observer/dependency"],
    }
    # The extra edge makes the observer run after the mutation deterministically.
    fanout["pipeline"]["outputs"] = {"observer": ["result"], "consumer": ["result"]}
    write("fanout", fanout)

    # Exercise NativeRT and TorchScript in the same process.
    pt2 = copy.deepcopy(mixed)
    torch.export.save(
        torch.export.export(Integer(), (values,)), root / "preprocess.pt2"
    )
    pt2["models"]["preprocess"]["parameters"] = {
        "backend": "pt2",
        "model_path": "preprocess.pt2",
    }
    write("mixed_pt2", pt2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
