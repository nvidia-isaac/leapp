#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import unittest
import torch
from leapp import annotate
from .base import LEAPPFunctionalTestBase


# Module-level constant for test_global_constant
SCALE_FACTOR = torch.tensor([2.0, 3.0, 4.0])


class TestLegacyMethod(LEAPPFunctionalTestBase):
    """Functional tests for annotate._method() — the legacy sys.settrace
    + ModuleBuilder decorator kept for internal use."""

    # ── Test 1a: Decorator syntax ────────────────────────────────────────

    def test_decorator_basic(self):
        """Wrapping a function with @annotate._method() decorator."""
        @annotate._method(export_with="jit")
        def double(x: torch.Tensor):
            return x * 2.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        for _ in range(5):
            out = double(torch.tensor([1.0, 2.0, 3.0]))
        annotate.stop()
        annotate.compile_graph(visualize=False, validate=True)

        self.assertEqual(len(annotate.nodes), 1)
        self.verify_all_models_exist("double")
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1.0, 2.0, 3.0])],
            [torch.tensor([2.0, 4.0, 6.0])],
            "double",
        )

    # ── Test 1b: Direct call syntax ──────────────────────────────────────

    def test_direct_call(self):
        """Using annotate._method()(func) directly instead of as a decorator."""
        def triple(x: torch.Tensor):
            return x * 3.0

        wrapped = annotate._method(export_with="jit")(triple)

        annotate.start(name=self.TEST_GRAPH_NAME)
        for _ in range(3):
            out = wrapped(torch.tensor([1.0, 2.0, 3.0]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.verify_all_models_exist("triple")
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1.0, 2.0, 3.0])],
            [torch.tensor([3.0, 6.0, 9.0])],
            "triple",
        )

    # ── Test 2: Chaining two _method() functions into a pipeline ─────────

    def test_chain_functions_pipeline(self):
        """Chain two _method() functions and verify pipeline connections."""
        @annotate._method(export_with="jit")
        def step_a(x: torch.Tensor):
            return x + 1.0

        @annotate._method(export_with="jit")
        def step_b(y: torch.Tensor):
            return y * 10.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        for _ in range(3):
            mid = step_a(torch.tensor([1.0, 2.0, 3.0]))
            out = step_b(mid)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 2)
        self.verify_all_models_exist("step_a", "step_b")
        self.verify_num_connections(annotate, nodes=2,
                                    internal_connections=1)

    # ── Test 3: Feedback into a function ─────────────────────────────────

    def test_feedback_into_function(self):
        """A function whose output is fed back as input on the next
        iteration should produce a feedback connection."""
        @annotate._method(export_with="jit")
        def accumulate(state: torch.Tensor):
            return state + 1.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        state = torch.zeros(3)
        for _ in range(5):
            state = accumulate(state)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.verify_all_models_exist("accumulate")
        self.verify_num_connections(annotate, feedback_connections=1)

    # ── Test 4: Global constant via environment_constants ────────────────

    def test_global_constant(self):
        """A function that references a global constant can still export
        when the constant is declared via environment_constants."""
        @annotate._method(export_with="jit",
                          environment_constants=["SCALE_FACTOR"])
        def scale(x: torch.Tensor):
            return x * SCALE_FACTOR

        annotate.start(name=self.TEST_GRAPH_NAME)
        for _ in range(3):
            out = scale(torch.tensor([1.0, 1.0, 1.0]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.verify_all_models_exist("scale")
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1.0, 1.0, 1.0])],
            [torch.tensor([2.0, 3.0, 4.0])],
            "scale",
        )

    # ── Test 5: Nested annotated functions → error ───────────────────────

    def test_nested_annotated_functions_error(self):
        """Calling one _method()-annotated function from inside another
        should fail because the tracing lock is already acquired."""
        @annotate._method(export_with="jit")
        def inner(x: torch.Tensor):
            return x * 2.0

        @annotate._method(export_with="jit")
        def outer(x: torch.Tensor):
            return inner(x)

        annotate.start(name=self.TEST_GRAPH_NAME)
        try:
            outer(torch.tensor([1.0, 2.0, 3.0]))
            annotate.stop()
            self.fail("Expected an exception for nested _method() calls")
        except Exception as e:
            annotate.stop()
            self.assertIn("Tracing lock is already acquired", str(e))

    # ── Test 6: TracedTensor as input to _method() → error ───────────────

    def test_traced_data_as_input_error(self):
        """Passing an active TracedTensor (from input_tensors) as an
        argument to a _method()-decorated function should raise an error."""
        @annotate._method(export_with="jit")
        def func(x: torch.Tensor):
            return x + 5.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        try:
            traced = annotate.input_tensors("other_node",
                                            {"val": torch.tensor([1.0, 2.0, 3.0])})
            traced = traced + 100.0
            func(traced)
            annotate.output_tensors("other_node", {"out": traced}, export_with="jit")
            annotate.stop()
            self.fail("Expected an exception for TracedTensor input to _method()")
        except Exception as e:
            annotate.stop()
            self.assertIn("Cannot use TracedTensor", str(e))

    # ── Test 7: Traced tensor ops while _method() is active → error ──────

    def test_traced_operation_while_annotated_error(self):
        """Creating traced tensors (via input_tensors) while inside a
        _method()-decorated function should fail because the tracing
        lock prevents mixing contexts."""
        @annotate._method(export_with="jit")
        def func(x: torch.Tensor):
            annotate.input_tensors("rogue_node",
                                   {"y": torch.randn(3)})
            return x * 2.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        try:
            func(torch.tensor([1.0, 2.0, 3.0]))
            annotate.stop()
            self.fail("Expected an exception for traced ops inside _method()")
        except Exception as e:
            annotate.stop()
            self.assertIn("Mixing active contxts is not allowed", str(e))


    # ── Test 8: Dict input → list output ────────────────────────────────

    def test_dict_input_list_output(self):
        """Dict input flattened to 3 tensors, returned as a list of 3."""
        @annotate._method(export_with="jit")
        def fuse(sensors: dict):
            return list(sensors.values())

        input_dict = {
            'lidar': torch.tensor([1.0, 2.0]),
            'camera': torch.tensor([3.0, 4.0]),
            'imu': torch.tensor([5.0, 6.0]),
        }

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected = fuse(input_dict)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        node = annotate.nodes["fuse"]
        self.assertEqual(len(node.inputs), 3)
        self.assertEqual(len(node.outputs), 3)
        self.verify_all_models_exist("fuse")
        self.verify_single_torchscript_model_expected_value(
            [input_dict], [expected], "fuse")

    # ── Test 9: List input → single output ───────────────────────────────

    def test_list_input_single_output(self):
        """List input flattened to 3 tensors, reduced to a single output."""
        @annotate._method(export_with="jit")
        def sum_all(items: list):
            return items[0] + items[1] + items[2]

        input_list = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
        ]

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected = sum_all(input_list)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        node = annotate.nodes["sum_all"]
        self.assertEqual(len(node.inputs), 3)
        self.assertEqual(len(node.outputs), 1)
        self.verify_all_models_exist("sum_all")
        self.verify_single_torchscript_model_expected_value(
            [input_list], [expected], "sum_all")

    # ── Test 10: Dict + list inputs → dict output ────────────────────────

    def test_dict_and_list_inputs_dict_output(self):
        """Mixed dict and list args producing a dict return value.
        4 flat input tensors, 2 flat output tensors."""
        @annotate._method(export_with="jit")
        def combine(state: dict, commands: list):
            pos = state['pos'] + commands[0]
            vel = state['vel'] + commands[1]
            return {'pos': pos, 'vel': vel}

        state = {'pos': torch.tensor([1.0, 2.0]), 'vel': torch.tensor([0.1, 0.2])}
        commands = [torch.tensor([0.5, 0.5]), torch.tensor([0.01, 0.02])]

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected = combine(state, commands)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        node = annotate.nodes["combine"]
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(len(node.outputs), 2)
        self.verify_all_models_exist("combine")
        self.verify_single_torchscript_model_expected_value(
            [state, commands], [expected], "combine")

    # ── Test 11: Nested dict input → single output ───────────────────────

    def test_nested_dict_input(self):
        """Nested dict-of-dicts flattened to 3 leaf tensors, single output."""
        @annotate._method(export_with="jit")
        def process(data: dict):
            a = data['group_a']['x']
            b = data['group_a']['y']
            c = data['group_b']['z']
            return a + b + c

        nested = {
            'group_a': {'x': torch.tensor([1.0]), 'y': torch.tensor([2.0])},
            'group_b': {'z': torch.tensor([3.0])},
        }

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected = process(nested)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        node = annotate.nodes["process"]
        self.assertEqual(len(node.inputs), 3)
        self.assertEqual(len(node.outputs), 1)
        self.verify_all_models_exist("process")
        self.verify_single_torchscript_model_expected_value(
            [nested], [expected], "process")

    # ── Test 12: Dict input → dict output (bidirectional) ────────────────

    def test_dict_input_dict_output(self):
        """Dict in, dict out — both sides flatten to the same leaf count."""
        @annotate._method(export_with="jit")
        def transform(sensors: dict):
            return {
                'scaled_lidar': sensors['lidar'] * 2.0,
                'scaled_camera': sensors['camera'] * 3.0,
            }

        input_dict = {
            'lidar': torch.tensor([1.0, 2.0]),
            'camera': torch.tensor([3.0, 4.0]),
        }

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected = transform(input_dict)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        node = annotate.nodes["transform"]
        self.assertEqual(len(node.inputs), 2)
        self.assertEqual(len(node.outputs), 2)
        self.verify_all_models_exist("transform")
        self.verify_single_torchscript_model_expected_value(
            [input_dict], [expected], "transform")

    # ── Test 13: List input → list output ────────────────────────────────

    def test_list_input_list_output(self):
        """List in, list out — element-wise transform preserving count."""
        @annotate._method(export_with="jit")
        def per_element(items: list):
            return [items[0] + 10.0, items[1] + 20.0, items[2] + 30.0]

        input_list = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
        ]

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected = per_element(input_list)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        node = annotate.nodes["per_element"]
        self.assertEqual(len(node.inputs), 3)
        self.assertEqual(len(node.outputs), 3)
        self.verify_all_models_exist("per_element")
        self.verify_single_torchscript_model_expected_value(
            [input_list], [expected], "per_element")

    # ── Test 14: Dict + list inputs → list + dict outputs ────────────────

    def test_bidirectional_complex_io(self):
        """Dict and list inputs producing both list and dict outputs.
        Verifies all 8 leaf tensors (4 in, 4 out) are handled."""
        @annotate._method(export_with="jit")
        def cross(state: dict, cmds: list):
            list_out = [state['a'] + cmds[0], state['b'] + cmds[1]]
            dict_out = {'sum_a': state['a'] + cmds[0], 'sum_b': state['b'] + cmds[1]}
            return list_out, dict_out

        state = {'a': torch.tensor([1.0]), 'b': torch.tensor([2.0])}
        cmds = [torch.tensor([10.0]), torch.tensor([20.0])]

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected = cross(state, cmds)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        node = annotate.nodes["cross"]
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(len(node.outputs), 4)
        self.verify_all_models_exist("cross")
        self.verify_single_torchscript_model_expected_value(
            [state, cmds], [expected], "cross")


if __name__ == "__main__":
    unittest.main(verbosity=2)
