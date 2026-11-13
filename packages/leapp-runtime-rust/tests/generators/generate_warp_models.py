# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture one Warp graph and export ONNX/PT2 bundles for plugin integration."""

import argparse
import json
from pathlib import Path

import torch
import warp as wp

from leapp.leapp_graph.custom_operator_registry.warp_operator.bundle import pack_bundle
from leapp.leapp_graph.custom_operator_registry.warp_operator.metadata import (
    build_runtime_metadata,
    encode_runtime_metadata,
)
from leapp.leapp_graph.datatypes.warp.warp_segment import WarpTensorRef


@wp.kernel
def transform(
    x: wp.array(dtype=wp.vec3),
    gain: wp.array(dtype=wp.int32),
    y: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    y[i] = x[i] * float(gain[i]) + wp.vec3(3.0)


class WarpModel(torch.nn.Module):
    def __init__(self, archive, metadata):
        super().__init__()
        self.register_buffer("archive", torch.tensor(list(archive), dtype=torch.uint8))
        self.metadata = metadata

    def forward(self, x, gain):
        return (
            torch.ops.leapp.warp_runner([x, gain], self.metadata, self.archive)[0] + 1
        )


def write_bundle(root, name, backend, model_path):
    def port(name, dtype, shape):
        return {"name": name, "dtype": dtype, "shape": shape, "type": "tensor"}

    bundle = {
        "system information": {"leapp config version": "1.3"},
        "models": {
            "model": {
                "inputs": [port("x", "float32", [4, 3]), port("gain", "int32", [4])],
                "outputs": [port("result", "float32", [4, 3])],
                "parameters": {"backend": backend, "model_path": str(model_path)},
            }
        },
        "pipeline": {
            "data_flow": {},
            "feedback_flow": {},
            "inputs": {"model": ["x", "gain"]},
            "outputs": {"model": ["result"]},
            "configs": {},
        },
    }
    (root / f"{name}.yaml").write_text(json.dumps(bundle))


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    wp.init()
    x = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
    gain = wp.zeros(4, dtype=wp.int32, device="cuda:0")
    y = wp.zeros_like(x)
    wp.load_module(device="cuda:0")
    with wp.ScopedCapture(device="cuda:0", apic=True) as capture:
        wp.launch(transform, dim=4, inputs=[x, gain], outputs=[y], device="cuda:0")
    wp.capture_save(
        capture.graph,
        str(root / "segment"),
        inputs={"x": x, "gain": gain},
        outputs={"y": y},
    )
    archive = pack_bundle(root / "segment.wrp")
    output = WarpTensorRef.from_value("y", y)
    metadata = encode_runtime_metadata(
        build_runtime_metadata(
            segment=None,
            input_refs=[
                WarpTensorRef.from_value("x", x),
                WarpTensorRef.from_value("gain", gain),
            ],
            output_refs=[output],
            output_shapes=[list(output.storage_shape)],
            output_dtypes=[output.storage_dtype],
            output_mask=[True],
        )
    )
    for device in ("cpu", "cuda"):
        args = (
            torch.zeros((4, 3), device=device),
            torch.ones(4, dtype=torch.int32, device=device),
        )
        program = torch.export.export(WarpModel(archive, metadata).to(device), args)
        name = f"warp_pt2_{device}"
        torch.export.save(program, root / f"{name}.pt2")
        write_bundle(root, name, "pt2", f"{name}.pt2")
        if device == "cpu":
            torch.onnx.export(
                program,
                args,
                root / "warp.onnx",
                input_names=["x", "gain"],
                output_names=["result"],
                opset_version=18,
                dynamo=True,
                external_data=False,
            )
            write_bundle(root, "warp_onnx", "onnx", "warp.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    generate(parser.parse_args().output)
