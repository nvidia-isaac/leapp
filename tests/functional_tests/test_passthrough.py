#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import torch
import numpy as np
import leapp
from leapp.leapp import _MANAGER as annotate
from .base import LEAPPFunctionalTestBase
from leapp.leapp_graph.datatypes import TracedTensor, TracedNpArray


class TestPassthrough(LEAPPFunctionalTestBase):

    def test_method_passthrough(self):
        seen_traced_inputs = []

        @annotate.method(export_with="jit")
        def policy(x: torch.Tensor):
            seen_traced_inputs.append(isinstance(x, TracedTensor))
            y = x * 2.0 + 1.0
            return y

        x = torch.tensor([1.0, 2.0, 3.0])
        y = policy(x)

        self.assertEqual(len(seen_traced_inputs), 1)
        self.assertFalse(seen_traced_inputs[0], "Input should remain a raw tensor outside tracing")
        self.assertNotIsInstance(y, TracedTensor)
        self.assertEqual(len(annotate.nodes), 0, "No nodes should be created outside start/stop")

    def test_input_output_tensor_passthrough(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([0.5, 0.5, 0.5])

        traced_a, traced_b = annotate.input_tensors("node_a", {"a": a, "b": b})
        self.assertIs(traced_a, a)
        self.assertIs(traced_b, b)
        self.assertNotIsInstance(traced_a, TracedTensor)
        self.assertNotIsInstance(traced_b, TracedTensor)

        out = traced_a + traced_b
        returned_out = annotate.output_tensors("node_a", {"out": out}, export_with="jit")
        self.assertIs(returned_out, out)

        self.assertNotIsInstance(out, TracedTensor)
        self.assertEqual(len(annotate.nodes), 0, "No nodes should be created outside start/stop")

    def test_state_tensor_passthrough(self):
        obs = torch.tensor([1.0, 2.0, 3.0])
        state = torch.tensor([0.1, 0.2, 0.3])

        passthrough_state = annotate.state_tensors("node_state", {"state": state})
        self.assertIs(passthrough_state, state)
        self.assertNotIsInstance(passthrough_state, TracedTensor)

        new_state = passthrough_state + obs
        returned_state = annotate.update_state("node_state", {"state": new_state})
        self.assertIs(returned_state, new_state)

        self.assertNotIsInstance(new_state, TracedTensor)
        self.assertEqual(len(annotate.nodes), 0, "No nodes should be created outside start/stop")

    def test_module_passthrough(self):
        class TinyStatefulModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("running_state", torch.zeros(4))

            def forward(self, x):
                return x + self.running_state

        model = TinyStatefulModel()
        original_buffers = dict(model.named_buffers())
        original_buffer_ids = {name: id(buf) for name, buf in original_buffers.items()}

        # Should no-op outside tracing, even without creating a LEAPP node.
        annotate.module("policy", model)

        current_buffers = dict(model.named_buffers())
        self.assertEqual(set(original_buffers.keys()), set(current_buffers.keys()))
        for name, buf in current_buffers.items():
            self.assertNotIsInstance(buf, TracedTensor)
            self.assertEqual(
                id(buf), original_buffer_ids[name],
                f"Buffer '{name}' should not be replaced outside tracing"
            )
        self.assertEqual(len(annotate.nodes), 0, "No nodes should be created outside start/stop")


class TestPersistentCarrier(LEAPPFunctionalTestBase):

    def test_numpy_output_crosses_node_boundary(self):
        leapp.start(name=self.TEST_GRAPH_NAME)

        source_input = annotate.input_tensors(
            "numpy_source",
            {"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        )
        source_output = source_input * 2.0
        annotate.output_tensors("numpy_source", {"y": source_output})

        self.assertIsInstance(source_output, TracedNpArray)
        self.assertFalse(source_output.is_tracing)
        self.assertEqual(source_output.output_port, "y")
        self.assertIs(source_output.context_obj, annotate.nodes["numpy_source"])

        consumer_input = annotate.input_tensors(
            "numpy_consumer", {"y": source_output}
        )
        self.assertIsInstance(consumer_input, TracedNpArray)
        self.assertIsNot(consumer_input, source_output)
        self.assertTrue(consumer_input.is_tracing)
        self.assertEqual(consumer_input.context, "numpy_consumer")
        self.assertEqual(source_output.context, "numpy_source")
        self.assertTrue(np.shares_memory(consumer_input.data, source_output.data))

        annotate.output_tensors(
            "numpy_consumer", {"z": consumer_input + 1.0}
        )
        leapp.stop()


class TestDryrun(LEAPPFunctionalTestBase):

    def test_complex_nested_input_and_state_tensors_dryrun_carriers(self):
        """dry_run should promote nested inputs, while nested states are rejected."""
        leapp.start(name=self.TEST_GRAPH_NAME, dry_run=True)

        nested_cases = [
            {
                "a": torch.tensor([1.0, 2.0]),
                "b": [torch.tensor([[3.0, 4.0]]), {"c": torch.tensor([5.0])}],
            },
            (
                torch.tensor([6.0]),
                {"d": (torch.tensor([7.0, 8.0]), [torch.tensor([9.0])])},
            ),
            [{"e": torch.tensor([10.0])}, (torch.tensor([11.0]), torch.tensor([12.0]))],
        ]

        def _assert_nested_passthrough(original, returned):
            original_tensors = self._flatten_to_tensors(original)
            returned_tensors = self._flatten_to_tensors(returned)
            self.assertEqual(len(original_tensors), len(returned_tensors))
            for original_tensor, returned_tensor in zip(original_tensors, returned_tensors):
                # Declaring promotes the caller's tensor in place, so the
                # returned carrier is that same object at every nesting depth.
                self.assertIs(original_tensor, returned_tensor)
                self.assertIsInstance(returned_tensor, TracedTensor)
                self.assertTrue(returned_tensor.is_tracing)
                self.assertTrue(torch.equal(original_tensor, returned_tensor.tensor))

        returned_inputs = []
        for idx, payload in enumerate(nested_cases):
            returned_input = annotate.input_tensors("complex_node", {f"in_{idx}": payload})
            _assert_nested_passthrough(payload, returned_input)
            returned_inputs.append(returned_input)
            with self.assertRaises(TypeError) as exc:
                annotate.state_tensors("complex_node", {f"state_{idx}": payload})
            self.assertIn("does not support nested state structures", str(exc.exception))

        annotate.output_tensors(
            "complex_node",
            {"out": {"inputs": returned_inputs}},
            export_with="jit",
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "complex_node.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "complex_node.onnx")))

    def test_method_dryrun_carrier_and_yaml(self):
        """dry_run should use inactive traced carriers and still produce graph YAML."""
        seen_traced_inputs = []

        @annotate.method(export_with="jit")
        def policy(x: torch.Tensor):
            seen_traced_inputs.append(isinstance(x, TracedTensor))
            y = x * 2.0 + 1.0
            return y

        leapp.start(name=self.TEST_GRAPH_NAME, dry_run=True)
        x = torch.tensor([1.0, 2.0, 3.0])
        y = policy(x)
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertEqual(len(seen_traced_inputs), 1)
        self.assertTrue(seen_traced_inputs[0])
        self.assertIsInstance(y, TracedTensor)
        self.assertFalse(y.is_tracing)
        self.assertEqual(y.output_port, "y",
                         "Dryrun should still publish output ports for connections")

        self.assertTrue(hasattr(annotate, "detected_pipeline"))
        self.assertTrue(hasattr(annotate, "detected_nodes"))
        self.assertIn("policy", annotate.detected_nodes)
        self.assertTrue(
            torch.is_tensor(y),
            "Method output should remain tensor-compatible in dryrun",
        )
        self.assertTrue(
            torch.allclose(y, x * 2.0 + 1.0),
            "Dryrun method output should preserve functional behavior",
        )
        self.assertTrue(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")))
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "policy.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "policy.onnx")))

    def test_input_output_tensor_dryrun_passthrough(self):
        """dry_run at start should still build connectivity but skip model export."""
        leapp.start(name=self.TEST_GRAPH_NAME, dry_run=True)

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([0.5, 0.5, 0.5])
        traced_a, traced_b = annotate.input_tensors("node_a", {"a": a, "b": b})
        self.assertIsInstance(traced_a, TracedTensor)
        self.assertIsInstance(traced_b, TracedTensor)
        self.assertTrue(traced_a.is_tracing)
        self.assertTrue(traced_b.is_tracing)

        out = traced_a + traced_b
        returned_out = annotate.output_tensors("node_a", {"out": out}, export_with="jit")
        self.assertIs(returned_out, out)

        downstream_in = annotate.input_tensors("node_b", {"out": out})
        self.assertIsInstance(downstream_in, TracedTensor)
        self.assertTrue(downstream_in.is_tracing)
        annotate.output_tensors("node_b", {"final": downstream_in * 3.0}, export_with="jit")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertEqual(out.output_port, "out",
                         "Dryrun should publish node output ports")
        self.assertIn("data_flow", annotate.detected_pipeline)
        self.assertEqual(1, len(annotate.detected_pipeline["data_flow"]))
        self.assertIn("node_a/out", annotate.detected_pipeline["data_flow"])
        self.assertEqual(["node_b/out"], list(annotate.detected_pipeline["data_flow"]["node_a/out"]))
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "node_a.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "node_b.pt")))

    def test_state_tensor_dryrun_passthrough(self):
        """dry_run at start should preserve feedback detection and skip export."""
        leapp.start(name=self.TEST_GRAPH_NAME, dry_run=True)

        obs = annotate.input_tensors("node_state", {"obs": torch.tensor([1.0, 2.0, 3.0])})
        state = annotate.state_tensors("node_state", {"state": torch.tensor([0.1, 0.2, 0.3])})
        self.assertIsInstance(obs, TracedTensor)
        self.assertIsInstance(state, TracedTensor)
        self.assertTrue(obs.is_tracing)
        self.assertTrue(state.is_tracing)

        new_state = state + obs
        returned_state = annotate.update_state("node_state", {"state": new_state})
        self.assertIs(returned_state, new_state)
        annotate.output_tensors("node_state", {"action": obs + state}, export_with="jit")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertIsInstance(new_state, TracedTensor)
        self.assertFalse(new_state.is_tracing)
        self.assertTrue(
            len(annotate.detected_pipeline["feedback_flow"]) >= 1,
            "Dryrun state path should still populate feedback_flow",
        )
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "node_state.pt")))

    def test_module_dryrun_passthrough(self):
        """dry_run at start allows module tracking but should restore buffers and skip export."""
        class TinyStatefulModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("running_state", torch.zeros(4))

            def forward(self, x):
                return x + self.running_state

        model = TinyStatefulModel()
        original_buffers = dict(model.named_buffers())
        original_buffer_ids = {name: id(buf) for name, buf in original_buffers.items()}

        leapp.start(name=self.TEST_GRAPH_NAME, dry_run=True)
        x = annotate.input_tensors("policy", {"x": torch.ones(4)})
        annotate.module("policy", model)
        y = model(x)
        annotate.output_tensors("policy", {"y": y}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        current_buffers = dict(model.named_buffers())
        self.assertEqual(set(original_buffers.keys()), set(current_buffers.keys()))
        for name, buf in current_buffers.items():
            self.assertNotIsInstance(buf, TracedTensor)
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "policy.pt")))

    def test_non_traced_node_is_selective_and_preserves_connectivity(self):
        """non_traced should skip export only for selected nodes while preserving graph connectivity."""
        leapp.start(name=self.TEST_GRAPH_NAME, non_traced=["raw_node"])

        x = torch.tensor([1.0, 2.0, 3.0])
        raw_x = annotate.input_tensors("raw_node", {"x": x})
        self.assertIsInstance(raw_x, TracedTensor)
        self.assertTrue(raw_x.is_tracing)

        raw_y = raw_x * 2.0
        annotate.output_tensors("raw_node", {"y": raw_y}, export_with="jit")
        self.assertEqual(raw_y.output_port, "y",
                         "non_traced outputs should still publish their port")

        traced_y = annotate.input_tensors("traced_node", {"y": raw_y})
        self.assertIsInstance(traced_y, TracedTensor)

        traced_z = traced_y + 1.0
        annotate.output_tensors("traced_node", {"z": traced_z}, export_with="jit")

        leapp.stop()
        results = leapp.compile_graph(visualize=False, validate=True)

        self.assertTrue(results["raw_node"])
        self.assertTrue(results["traced_node"])
        self.assertIn("raw_node", annotate.detected_nodes)
        self.assertIn("traced_node", annotate.detected_nodes)
        self.assertTrue(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "traced_node.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "raw_node.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "raw_node.onnx")))
        self.assertIn("raw_node/y", annotate.detected_pipeline["data_flow"])
        self.assertEqual(
            ["traced_node/y"],
            list(annotate.detected_pipeline["data_flow"]["raw_node/y"]),
        )

    def test_non_traced_node_validation_is_skipped_under_validate_true(self):
        """validate=True should succeed when a graph mixes traced and non_traced nodes."""
        leapp.start(name=self.TEST_GRAPH_NAME, non_traced=["raw_node"])

        raw_x = annotate.input_tensors("raw_node", {"x": torch.tensor([2.0, 4.0, 6.0])})
        raw_y = raw_x / 2.0
        annotate.output_tensors("raw_node", {"y": raw_y}, export_with="jit")

        traced_y = annotate.input_tensors("traced_node", {"y": raw_y})
        annotate.output_tensors("traced_node", {"z": traced_y.square()}, export_with="jit")

        leapp.stop()
        results = leapp.compile_graph(visualize=False, validate=True)

        self.assertEqual({"raw_node", "traced_node"}, set(results.keys()))
        self.assertTrue(results["raw_node"])
        self.assertTrue(results["traced_node"])
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "raw_node.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "traced_node.pt")))

    def test_non_traced_state_node_still_populates_feedback_flow(self):
        """non_traced state nodes should still register feedback connections without exporting a model."""
        leapp.start(name=self.TEST_GRAPH_NAME, non_traced=["state_node"])

        obs = annotate.input_tensors("state_node", {"obs": torch.tensor([1.0, 2.0, 3.0])})
        state = annotate.state_tensors("state_node", {"state": torch.tensor([0.1, 0.2, 0.3])})
        self.assertIsInstance(obs, TracedTensor)
        self.assertIsInstance(state, TracedTensor)
        self.assertTrue(obs.is_tracing)
        self.assertTrue(state.is_tracing)

        new_state = state + obs
        annotate.update_state("state_node", {"state": new_state})
        annotate.output_tensors("state_node", {"action": obs - state}, export_with="jit")

        leapp.stop()
        results = leapp.compile_graph(visualize=False, validate=True)

        self.assertTrue(results["state_node"])
        self.assertIn("feedback_flow", annotate.detected_pipeline)
        self.assertGreaterEqual(len(annotate.detected_pipeline["feedback_flow"]), 1)
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "state_node.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.TEST_GRAPH_NAME, "state_node.onnx")))
