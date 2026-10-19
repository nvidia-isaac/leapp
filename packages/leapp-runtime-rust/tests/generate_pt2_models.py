# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export PT2 programs and model metadata for native tests."""

import argparse
import json
from pathlib import Path

import torch


class Weighted(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2., 3., 4., 5.], device=device))
        self.register_buffer("bias", torch.tensor([10., 20., 30., 40.], device=device))

    def forward(self, values):
        return values * self.weight + self.bias, values.reshape(2, 2).t()


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for device in devices:
        suffix = "cuda" if device.startswith("cuda") else "cpu"
        module = Weighted(device).eval()
        example = torch.arange(1., 5., device=device)
        directory = root / f"weighted_{suffix}"
        directory.mkdir(exist_ok=True)
        program = torch.export.export(module, (example.clone(),))
        torch.export.save(program, directory / "model.pt2")
        expected = module(example.clone())
        dtype = str(example.dtype).removeprefix("torch.")
        inputs = [{"name": "values", "dtype": dtype,
                   "shape": list(example.shape),
                   "type": "tensor"}]
        outputs = [
            {"name": f"output_{i}",
             "dtype": str(value.dtype).removeprefix("torch."),
             "shape": list(value.shape),
             "type": "tensor"}
            for i, value in enumerate(expected)
        ]
        bundle = {
            "system information": {"leapp config version": "1.3"},
            "models": {"model": {
                "inputs": inputs, "outputs": outputs,
                "parameters": {"backend": "pt2",
                               "model_path": f"{directory.name}/model.pt2"},
            }},
            "pipeline": {"data_flow": {}, "feedback_flow": {},
                         "inputs": {"model": ["values"]},
                         "outputs": {"model": [p["name"] for p in outputs]},
                         "configs": {}},
        }
        (root / f"{directory.name}.yaml").write_text(json.dumps(bundle))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    generate(parser.parse_args().output)
