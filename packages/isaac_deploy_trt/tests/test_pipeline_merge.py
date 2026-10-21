#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for LEAPP ONNX pipeline merge (TensorRTNode, no Triton)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
import yaml

from isaac_deploy_trt.cli import main
from isaac_deploy_trt.pipeline_merge import (
    merge_leapp_onnx_pipeline,
    save_merged_pipeline,
)


def _abs_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=[helper.make_node("Abs", ["x"], ["h"], name="abs0")],
        name="abs",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("h", TensorProto.FLOAT, [2])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _relu_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=[helper.make_node("Relu", ["h"], ["y"], name="relu0")],
        name="relu",
        inputs=[helper.make_tensor_value_info("h", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _identity_model(inp: str, out: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", [inp], [out], name=f"id_{out}")],
        name="id",
        inputs=[helper.make_tensor_value_info(inp, TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info(out, TensorProto.FLOAT, [2])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _write_chain_export(tmp: str) -> str:
    onnx.save(_abs_model(), os.path.join(tmp, "pre.onnx"))
    onnx.save(_relu_model(), os.path.join(tmp, "post.onnx"))
    spec = {
        "models": {
            "pre": {
                "inputs": [{"name": "x", "dtype": "float32", "shape": [2]}],
                "outputs": [{"name": "h", "dtype": "float32", "shape": [2]}],
                "parameters": {"model_path": "pre.onnx", "backend": "onnx"},
            },
            "post": {
                "inputs": [{"name": "h", "dtype": "float32", "shape": [2]}],
                "outputs": [{"name": "y", "dtype": "float32", "shape": [2]}],
                "parameters": {"model_path": "post.onnx", "backend": "onnx"},
            },
        },
        "pipeline": {
            "data_flow": {"pre/h": ["post/h"]},
            "feedback_flow": {},
            "inputs": {"pre": ["x"]},
            "outputs": {"post": ["y"]},
        },
    }
    path = os.path.join(tmp, "exported.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle)
    return path


class TestOnnxPipelineMerge(unittest.TestCase):
    def test_merge_abs_relu_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = _write_chain_export(tmp)
            result = merge_leapp_onnx_pipeline(yaml_path)
            self.assertEqual(result.input_tensor_names, ["x"])
            self.assertEqual(result.output_tensor_names, ["y"])
            ops = [node.op_type for node in result.model.graph.node]
            self.assertEqual(ops, ["Abs", "Relu"])
            onnx.checker.check_model(result.model)

            try:
                import onnxruntime as ort
            except ImportError:
                return
            session = ort.InferenceSession(
                result.model.SerializeToString(),
                providers=["CPUExecutionProvider"],
            )
            feed = np.array([-1.0, 2.0], dtype=np.float32)
            out = session.run(["y"], {"x": feed})[0]
            np.testing.assert_allclose(out, np.array([1.0, 2.0], dtype=np.float32))

    def test_fan_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            onnx.save(_abs_model(), os.path.join(tmp, "pre.onnx"))
            onnx.save(_relu_model(), os.path.join(tmp, "a.onnx"))
            onnx.save(_identity_model("h", "z"), os.path.join(tmp, "b.onnx"))
            spec = {
                "models": {
                    "pre": {
                        "inputs": [{"name": "x", "dtype": "float32", "shape": [2]}],
                        "outputs": [{"name": "h", "dtype": "float32", "shape": [2]}],
                        "parameters": {"model_path": "pre.onnx", "backend": "onnx"},
                    },
                    "a": {
                        "inputs": [{"name": "h", "dtype": "float32", "shape": [2]}],
                        "outputs": [{"name": "y", "dtype": "float32", "shape": [2]}],
                        "parameters": {"model_path": "a.onnx", "backend": "onnx"},
                    },
                    "b": {
                        "inputs": [{"name": "h", "dtype": "float32", "shape": [2]}],
                        "outputs": [{"name": "z", "dtype": "float32", "shape": [2]}],
                        "parameters": {"model_path": "b.onnx", "backend": "onnx"},
                    },
                },
                "pipeline": {
                    "data_flow": {"pre/h": ["a/h", "b/h"]},
                    "feedback_flow": {},
                    "inputs": {"pre": ["x"]},
                    "outputs": {"a": ["y"], "b": ["z"]},
                },
            }
            yaml_path = os.path.join(tmp, "exported.yaml")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(spec, handle)
            result = merge_leapp_onnx_pipeline(yaml_path)
            self.assertEqual(result.input_tensor_names, ["x"])
            self.assertEqual(result.output_tensor_names, ["y", "z"])
            onnx.checker.check_model(result.model)

    def test_input_output_name_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            onnx.save(_identity_model("left_arm", "h"), os.path.join(tmp, "pre.onnx"))
            onnx.save(_identity_model("h", "left_arm"), os.path.join(tmp, "post.onnx"))
            spec = {
                "models": {
                    "pre": {
                        "inputs": [{
                            "name": "left_arm",
                            "dtype": "float32",
                            "shape": [2],
                            "kind": "state/joint/position",
                            "element_names": [["joint_a", "joint_b"]],
                        }],
                        "outputs": [{"name": "h", "dtype": "float32", "shape": [2]}],
                        "parameters": {"model_path": "pre.onnx", "backend": "onnx"},
                    },
                    "post": {
                        "inputs": [{"name": "h", "dtype": "float32", "shape": [2]}],
                        "outputs": [{"name": "left_arm", "dtype": "float32", "shape": [2]}],
                        "parameters": {"model_path": "post.onnx", "backend": "onnx"},
                    },
                },
                "pipeline": {
                    "data_flow": {"pre/h": ["post/h"]},
                    "feedback_flow": {},
                    "inputs": {"pre": ["left_arm"]},
                    "outputs": {"post": ["left_arm"]},
                },
            }
            yaml_path = os.path.join(tmp, "exported.yaml")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(spec, handle)
            result = merge_leapp_onnx_pipeline(yaml_path)
            self.assertEqual(result.input_tensor_names, ["_in_left_arm"])
            self.assertEqual(result.output_tensor_names, ["left_arm"])
            self.assertEqual(result.input_rename, {"left_arm": "_in_left_arm"})
            merged_input = result.yaml_spec["models"]["pipeline"]["inputs"][0]
            self.assertEqual(merged_input["element_names"], [["joint_a", "joint_b"]])
            onnx.checker.check_model(result.model)

    def test_rewrite_during_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            resize = helper.make_node(
                "Resize",
                ["x", "", "", "scales"],
                ["h"],
                name="resize0",
                mode="linear",
                antialias=1,
            )
            scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 1, 1])
            graph = helper.make_graph(
                nodes=[resize],
                name="rz",
                inputs=[
                    helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
                ],
                outputs=[
                    helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, 1, 2, 2])
                ],
                initializer=[scales],
            )
            onnx.save(
                helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
                os.path.join(tmp, "pre.onnx"),
            )
            onnx.save(_relu_model(), os.path.join(tmp, "post.onnx"))
            spec = {
                "models": {
                    "pre": {
                        "inputs": [{"name": "x", "dtype": "float32", "shape": [1, 1, 2, 2]}],
                        "outputs": [{"name": "h", "dtype": "float32", "shape": [1, 1, 2, 2]}],
                        "parameters": {"model_path": "pre.onnx", "backend": "onnx"},
                    },
                    "post": {
                        "inputs": [{"name": "h", "dtype": "float32", "shape": [2]}],
                        "outputs": [{"name": "y", "dtype": "float32", "shape": [2]}],
                        "parameters": {"model_path": "post.onnx", "backend": "onnx"},
                    },
                },
                "pipeline": {
                    "data_flow": {"pre/h": ["post/h"]},
                    "feedback_flow": {},
                    "inputs": {"pre": ["x"]},
                    "outputs": {"post": ["y"]},
                },
            }
            yaml_path = os.path.join(tmp, "exported.yaml")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(spec, handle)
            result = merge_leapp_onnx_pipeline(yaml_path, rewrite=True)
            self.assertEqual(result.rewritten_models, ["pre"])
            resize_nodes = [
                node for node in result.model.graph.node if node.op_type == "Resize"
            ]
            self.assertEqual(len(resize_nodes), 1)
            antialias = next(
                (a.i for a in resize_nodes[0].attribute if a.name == "antialias"), 0
            )
            self.assertEqual(int(antialias), 0)

    def test_cli_merge_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = _write_chain_export(tmp)
            out = os.path.join(tmp, "pipeline.onnx")
            self.assertEqual(
                main(["merge-pipeline", yaml_path, "-o", out]),
                0,
            )
            self.assertTrue(os.path.isfile(out))
            sidecar = os.path.join(tmp, "pipeline.yaml")
            self.assertTrue(os.path.isfile(sidecar))
            with open(sidecar, encoding="utf-8") as handle:
                spec = yaml.safe_load(handle)
            self.assertEqual(spec["tensor_rt_node"]["input_tensor_names"], ["x"])
            self.assertEqual(spec["tensor_rt_node"]["output_tensor_names"], ["y"])
            onnx.checker.check_model(out)
            save_merged_pipeline(
                merge_leapp_onnx_pipeline(yaml_path),
                os.path.join(tmp, "again.onnx"),
            )


if __name__ == "__main__":
    unittest.main()
