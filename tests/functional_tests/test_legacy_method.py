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
        for _ in range(3):
            out = double(torch.tensor([1.0, 2.0, 3.0]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
