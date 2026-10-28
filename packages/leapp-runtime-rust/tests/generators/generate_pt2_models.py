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
        self.weight = torch.nn.Parameter(
            torch.tensor([2.0, 3.0, 4.0, 5.0], device=device)
        )
        self.register_buffer(
            "bias", torch.tensor([10.0, 20.0, 30.0, 40.0], device=device)
        )

    def forward(self, values):
        return values * self.weight + self.bias, values.reshape(2, 2).t()


class Mutating(torch.nn.Module):
    def forward(self, values):
        values.add_(2)
        return values * 3


class Reshape(torch.nn.Module):
    def forward(self, values):
        return values.reshape(-1, 2)


class Integer(torch.nn.Module):
    def forward(self, values):
        return values + 1


class Identity(torch.nn.Module):
    def forward(self, values):
        return values


class Stateful(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.register_buffer("total", torch.zeros(4, device=device))

    def forward(self, values):
        self.total.add_(values)
        return self.total.clone()


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for device in devices:
        suffix = "cuda" if device.startswith("cuda") else "cpu"
        for name, module, example, dynamic in [
            ("weighted", Weighted(device), torch.arange(1.0, 5.0, device=device), None),
            ("mutating", Mutating(), torch.arange(1.0, 5.0, device=device), None),
            (
                "reshape",
                Reshape(),
                torch.arange(4.0, device=device),
                {"values": {0: 2 * torch.export.Dim("half", min=1, max=16)}},
            ),
            ("integer", Integer(), torch.arange(4, device=device), None),
            ("scalar", Identity(), torch.tensor(3.0, device=device), None),
            ("empty", Identity(), torch.empty(0, device=device), None),
            (
                "double",
                Identity(),
                torch.arange(4, dtype=torch.float64, device=device),
                None,
            ),
            (
                "short",
                Identity(),
                torch.arange(4, dtype=torch.int16, device=device),
                None,
            ),
            ("stateful", Stateful(device), torch.ones(4, device=device), None),
        ]:
            module.eval()
            directory = root / f"{name}_{suffix}"
            directory.mkdir(exist_ok=True)
            program = torch.export.export(
                module, (example.clone(),), dynamic_shapes=dynamic
            )
            torch.export.save(program, directory / "model.pt2")
            expected = module(example.clone())
            if isinstance(expected, torch.Tensor):
                expected = (expected,)
            dtype = str(example.dtype).removeprefix("torch.")
            inputs = [
                {
                    "name": "values",
                    "dtype": dtype,
                    "shape": [-1] if dynamic else list(example.shape),
                    "type": "tensor",
                }
            ]
            outputs = [
                {
                    "name": f"output_{i}",
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "shape": [-1, 2] if dynamic else list(value.shape),
                    "type": "tensor",
                }
                for i, value in enumerate(expected)
            ]
            bundle = {
                "system information": {"leapp config version": "1.3"},
                "models": {
                    "model": {
                        "inputs": inputs,
                        "outputs": outputs,
                        "parameters": {
                            "backend": "pt2",
                            "model_path": f"{directory.name}/model.pt2",
                        },
                    }
                },
                "pipeline": {
                    "data_flow": {},
                    "feedback_flow": {},
                    "inputs": {"model": ["values"]},
                    "outputs": {"model": [p["name"] for p in outputs]},
                    "configs": {},
                },
            }
            (root / f"{directory.name}.yaml").write_text(json.dumps(bundle))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    generate(parser.parse_args().output)
