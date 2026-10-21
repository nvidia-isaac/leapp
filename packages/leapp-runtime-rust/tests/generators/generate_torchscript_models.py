# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export small TorchScript bundles and direct PyTorch reference outputs."""

import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    generate(parser.parse_args().output)
