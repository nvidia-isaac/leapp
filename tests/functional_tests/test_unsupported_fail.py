#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools
import unittest
import torch
import leapp
from leapp.leapp import _MANAGER as annotate
from leapp.utils.logging import _get_logger
from .base import LEAPPFunctionalTestBase


# ---------------------------------------------------------------------------
# Module-level helpers for caller identity tests
# ---------------------------------------------------------------------------

def _helper_path_a(name, tensor):
    return annotate.input_tensors(name, {'input': tensor})

def _helper_path_b(name, tensor):
    return annotate.input_tensors(name, {'input': tensor})

def _shared_helper(name, tensor):
    return annotate.input_tensors(name, {'input': tensor})

class _BasePipeline:
    def annotate_input(self, tensor):
        return annotate.input_tensors('func', {'input': tensor})

class _PipelineA(_BasePipeline):
    def run(self, tensor):
        return self.annotate_input(tensor)

class _PipelineB(_BasePipeline):
    def run(self, tensor):
        return self.annotate_input(tensor)

def _timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def _raw_annotate(tensor):
    return annotate.input_tensors('func', {'input': tensor})

_decorated_annotate = _timing_decorator(_raw_annotate)


class TestUnsupportedFail(LEAPPFunctionalTestBase):
    """Unit tests to see if unsupported io is properly handled"""

    def test_input_tensors_bare_tensor_fails(self):
        """Bare tensors must be wrapped in a dict or TensorSemantics."""
        with self.assertRaises(TypeError) as exc:
            annotate.input_tensors('func', torch.tensor([1.0, 2.0, 3.0]))
        self.assertIn("does not accept a bare tensor", str(exc.exception))

    def test_output_tensors_bare_tensor_fails(self):
        """Bare traced outputs must be wrapped in a dict or TensorSemantics."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        try:
            traced = annotate.input_tensors('func', {'input': torch.tensor([1.0, 2.0, 3.0])})
            with self.assertRaises(TypeError) as exc:
                annotate.output_tensors('func', traced + 1.0, export_with="jit")
            self.assertIn("does not accept a bare tensor", str(exc.exception))
        finally:
            leapp.stop()

    def test_static_outputs_bare_tensor_fails(self):
        """Bare static outputs must be wrapped in a dict or TensorSemantics."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        try:
            traced = annotate.input_tensors('func', {'input': torch.tensor([1.0, 2.0, 3.0])})
            with self.assertRaises(TypeError) as exc:
                annotate.output_tensors(
                    'func',
                    {'output': traced + 1.0},
                    static_outputs=torch.tensor([4.0, 5.0, 6.0]),
                    export_with="jit",
                )
            self.assertIn("does not accept a bare tensor", str(exc.exception))
        finally:
            leapp.stop()

    def test_input_tensors_top_level_raw_list_fails(self):
        """Top-level raw collections must be named via a dict."""
        with self.assertRaises(TypeError) as exc:
            annotate.input_tensors('func', [
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([4.0, 5.0, 6.0]),
            ])
        self.assertIn("expects either a dict of named tensors", str(exc.exception))

    def test_output_tensors_top_level_raw_list_fails(self):
        """Top-level raw output collections must be named via a dict."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        try:
            traced = annotate.input_tensors('func', {'input': torch.tensor([1.0, 2.0, 3.0])})
            outputs = [traced + 1.0, traced + 2.0]
            with self.assertRaises(TypeError) as exc:
                annotate.output_tensors('func', outputs, export_with="jit")
            self.assertIn("expects either a dict of named tensors", str(exc.exception))
        finally:
            leapp.stop()

    def test_validation_message_uses_sample_index(self):
        """Validation logs should label cached replay failures as sample N."""
        leapp.start(name=self.TEST_GRAPH_NAME, save_path=self.TEST_GRAPH_NAME)

        for value in (1.0, 2.0):
            traced = annotate.input_tensors('func', {'input': torch.tensor([value, value + 1.0])})
            annotate.output_tensors('func', {'output': traced + 1.0}, export_with="jit")

        leapp.stop()

        annotate.nodes['func'].outputs[0].cached_values[0] += 10.0
        results = leapp.compile_graph(visualize=False, validate=True, strict=False)

        self.assertFalse(results['func'])

        log_path = _get_logger().path
        with open(log_path) as f:
            log_text = f.read()

        self.assertIn("sample 1: Mismatch detected", log_text)
        self.assertNotIn("cached[0]", log_text)

    def test_validation_error_hint_when_only_sample_zero_passes(self):
        """Strict validation should hint at inlined constants when sample 0 passes but re-entry fails."""
        leapp.start(name=self.TEST_GRAPH_NAME, save_path=self.TEST_GRAPH_NAME)

        for value in (1.0, 2.0):
            traced = annotate.input_tensors('func', {'input': torch.tensor([value, value + 1.0])})
            annotate.output_tensors('func', {'output': traced + 1.0}, export_with="jit")

        leapp.stop()

        annotate.nodes['func'].outputs[0].cached_values[0] += 10.0

        with self.assertRaises(Exception) as exc:
            leapp.compile_graph(visualize=False, validate=True, strict=True)

        message = str(exc.exception)
        self.assertIn("Model validation failed", message)
        self.assertIn("inlined as a constant", message)
        self.assertIn("annotate.input_tensors()", message)

    def test_output_tensors_without_prior_input_tensors(self):
        """Calling output_tensors before input_tensors should raise a clear error."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        try:
            annotate.output_tensors(
                'orphan_node',
                {'output': torch.tensor([1.0, 2.0, 3.0])},
                export_with="jit",
            )
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn(
                "input_tensors() was never called for it",
                str(e),
            )
        finally:
            leapp.stop()

    def test_reentry_output_shape_change_fails(self):
        """Re-entering a node with a changed output shape should be rejected."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)

            for idx in range(2):
                traced = annotate.input_tensors(
                    'shape_node', {'input': torch.tensor([1.0, 2.0, 3.0])}
                )
                output = traced + 1.0 if idx == 0 else traced.reshape(1, 3)
                annotate.output_tensors(
                    'shape_node', {'output': output}, export_with="jit"
                )

            leapp.stop()
            self.fail("Expected an exception")
        except Exception as e:
            try:
                leapp.stop()
            except Exception:
                pass
            self.assertIn("Validation error when reentering node", str(e))

    def test_reentry_output_tag_change_fails(self):
        """Corrupting cached output tags before re-entry should be rejected."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)

            for idx in range(2):
                traced = annotate.input_tensors(
                    'tag_node', {'input': torch.tensor([1.0, 2.0, 3.0])}
                )
                annotate.output_tensors(
                    'tag_node', {'output': traced + 1.0}, export_with="jit"
                )
                if idx == 0:
                    annotate.nodes['tag_node'].outputs[0].tag = 'wrong_node/output/'

            leapp.stop()
            self.fail("Expected an exception")
        except Exception as e:
            try:
                leapp.stop()
            except Exception:
                pass
            self.assertIn("Validation error when reentering node", str(e))

    def test_same_variable_used_twice(self):

        @annotate.method()
        def funcA(inputA: torch.Tensor):
            return inputA

        @annotate.method()
        def funcB(inputB: torch.Tensor, inputC: torch.Tensor):
            return inputB+inputC

        leapp.start(name=self.TEST_GRAPH_NAME)
        retvalA = funcA(torch.tensor([1, 2, 3]))
        funcB(retvalA, retvalA)
        leapp.stop()

        try:
            leapp.compile_graph()
        except Exception as e:
            self.assertEqual(
                str(e), "Error: unsupported use of sending the same tensor multiple times to the same node")
            return
        self.fail("Expected an exception")

    def test_function_name_overlap(self):
        node_name = "funcA"

        @annotate.method(node_name=node_name)
        def func(inputA: torch.Tensor):
            return inputA

        @annotate.method(node_name=node_name)
        def func_copy(inputA: torch.Tensor):
            return inputA

        leapp.start(name=self.TEST_GRAPH_NAME)

        try:
            return_value = func(torch.tensor([1, 2, 3]))
            func_copy(return_value)
        except Exception as e:
            leapp.stop()
            expected = "Cannot reuse a node name from a different call site."
            self.assertIn(expected, str(e))
            return

        leapp.stop()
        self.fail("Expected an exception")

    def test_io_reconciliation_name_overlap(self):
        @annotate.method()
        def funcA(input: torch.Tensor):
            detections = annotate.register_buffer("funcA", torch.zeros(input.shape))
            # some processing
            return detections

        @annotate.method()
        def funcB(detections):
            retval = detections
            # some processing
            return retval

        @annotate.method()
        def funcC(input, detections):
            retval = input + detections
            return retval

        leapp.start(name=self.TEST_GRAPH_NAME)
        detections = funcA(torch.tensor([1.0, 2.0, 3.0]))
        funcB(detections)
        funcC(detections, torch.tensor([1.0, 2.0, 3.0]))
        leapp.stop()
        try:
            leapp.compile_graph()
        except Exception as e:
            self.assertEqual(str(e),
                             "Error requesting input name change for funcC/input: detections is already in use")

    def test_mirror_leapp_tags_data_mismatch(self):
        """Test mirror_leapp_tags with various data mismatch scenarios"""
        @annotate.method(export_with="jit")
        def funcA(inputA: torch.Tensor):
            return inputA * 2.0

        @annotate.method(export_with="jit")
        def funcB(inputA: torch.Tensor):
            return inputA, inputA + 1.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        
        # Test 1: Tensors with at least one element that is not equal
        # This should log an error but not raise an exception
        out_funcA = funcA(torch.tensor([1.0, 2.0, 3.0]))
        wrong_buffer = torch.tensor([1.0, 2.0, 4.0])  # Last element is wrong
        try:
            annotate.mirror_leapp_tags(out_funcA, wrong_buffer)  # Logs error, doesn't raise
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("source and target do not match", str(e))
        
        # Test 2: List of tensors with one element wrong
        # This should log an error but not raise an exception
        out1, out2 = funcB(torch.tensor([1.0, 2.0, 3.0]))
        source_list = [out1, out2]
        target_list = [out1.clone(), torch.tensor([1.0, 2.0, 99.0])]  # Second tensor is wrong
        try:
            annotate.mirror_leapp_tags(source_list, target_list)  # Logs error, doesn't raise
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("source and target do not match", str(e))
        
        # Test 3: Lists with different number of elements
        # This will cause an exception because mirror_all_tensor_tags will crash
        source_list2 = [out1, out2]
        target_list2 = [out1.clone()]  # Missing one element
        try:
            annotate.mirror_leapp_tags(source_list2, target_list2)
            self.fail("Expected an exception")
        except Exception as e:
            # Should get "unexpected error" due to index out of bounds
            self.assertIn("unexpected error", str(e))
        
        # Test 4: Dicts with different keys
        # This should log an error but not raise an exception
        source_dict = {'a': out_funcA, 'b': out_funcA + 1.0}
        target_dict = {'a': out_funcA.clone(), 'c': (out_funcA + 1.0).clone()}  # 'b' vs 'c'
        try:
            annotate.mirror_leapp_tags(source_dict, target_dict)  # Logs error, doesn't raise
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("source and target do not match", str(e))
        
        leapp.stop()

    def test_reentrant_tracing_method_inside_method(self):
        """Test that nested method decorators are properly rejected.
        
        Calling one @annotate.method from within another should fail
        because the tracing lock is already acquired.
        """
        @annotate.method()
        def inner_func(inputA: torch.Tensor):
            return inputA * 2.0

        @annotate.method()
        def outer_func(inputA: torch.Tensor):
            # This should fail - calling another annotated method while tracing
            result = inner_func(inputA)
            return result

        leapp.start(name=self.TEST_GRAPH_NAME)
        
        try:
            outer_func(torch.tensor([1.0, 2.0, 3.0]))
            leapp.stop()
            self.fail("Expected an exception for re-entrant tracing")
        except Exception as e:
            leapp.stop()
            # The error should mention that we're trying to set up a new trace
            self.assertIn("Mixing active contexts is not allowed", str(e))
    
    def test_reentrant_tracing_using_traced_tensors(self): #TODO: these both need to fail
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            tensors = annotate.input_tensors('func', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensors += 100
            input = torch.tensor([1.0, 2.0, 3.0])
            @annotate.method()
            def inner_func(inputA: torch.Tensor):
                return inputA + tensors
            output_tensors = inner_func(input)
            annotate.output_tensors('func', {'outputA': output_tensors}, export_with="jit")
            leapp.stop()
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("Mixing active contexts is not allowed", str(e))
    
    def test_passing_traced_tensor_to_method(self):
        try:
            @annotate.method()
            def inner_func(inputA: torch.Tensor):
                return inputA + 5
            leapp.start(name=self.TEST_GRAPH_NAME)
            tensors = annotate.input_tensors('func', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensors += 100
            output_tensors = inner_func(tensors)
            annotate.output_tensors('func', {'outputA': output_tensors}, export_with="jit")
            leapp.stop()
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("Mixing active contexts is not allowed", str(e))
    
    def test_cross_context_traced_tensor_usage(self):
        """Basic test: addition of two TracedTensors from different contexts."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([1.0, 2.0, 3.0])})
            output_tensors = tensor1 + tensor2
            annotate.output_tensors('func1', {'outputA': output_tensors}, export_with="jit")
            leapp.stop()
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("Mixing active contexts is not allowed", str(e))

    def test_cross_context_torch_cat_list(self):
        """Test torch.cat with a list containing TracedTensors from different contexts."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([4.0, 5.0, 6.0])})
            # torch.cat takes a list of tensors - should detect cross-context in the list
            output_tensors = torch.cat([tensor1, tensor2], dim=0)
            leapp.stop()
            self.fail("Expected an exception for cross-context torch.cat")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))

    def test_cross_context_torch_stack(self):
        """Test torch.stack with tensors from different contexts."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([4.0, 5.0, 6.0])})
            # torch.stack also takes a list
            output_tensors = torch.stack([tensor1, tensor2], dim=0)
            leapp.stop()
            self.fail("Expected an exception for cross-context torch.stack")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))

    def test_cross_context_matmul(self):
        """Test matrix multiplication with TracedTensors from different contexts."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([[5.0, 6.0], [7.0, 8.0]])})
            # Matrix multiplication with two explicit args
            output_tensors = torch.matmul(tensor1, tensor2)
            leapp.stop()
            self.fail("Expected an exception for cross-context torch.matmul")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))

    def test_cross_context_torch_add_with_alpha(self):
        """Test torch.add with alpha kwarg - mixing contexts in positional args."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([4.0, 5.0, 6.0])})
            # torch.add(input, other, alpha=1) - tests kwargs handling
            output_tensors = torch.add(tensor1, tensor2, alpha=2.0)
            leapp.stop()
            self.fail("Expected an exception for cross-context torch.add with alpha")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))

    def test_cross_context_three_contexts(self):
        """Test detection with three different contexts mixed together."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([4.0, 5.0, 6.0])})
            tensor3 = annotate.input_tensors('func3', {'inputC': torch.tensor([7.0, 8.0, 9.0])})
            # Mix all three contexts
            output_tensors = torch.cat([tensor1, tensor2, tensor3], dim=0)
            leapp.stop()
            self.fail("Expected an exception for three cross-contexts")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))

    def test_cross_context_torch_where(self):
        """Test torch.where with condition, x, y from different contexts."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            # condition from one context
            cond_tensor = annotate.input_tensors('cond_ctx', {'cond': torch.tensor([True, False, True])})
            # x from another context
            x_tensor = annotate.input_tensors('x_ctx', {'x': torch.tensor([1.0, 2.0, 3.0])})
            # y is a regular tensor (should be fine)
            y_tensor = torch.tensor([10.0, 20.0, 30.0])
            # torch.where(condition, x, y) - two TracedTensors from different contexts
            output_tensors = torch.where(cond_tensor, x_tensor, y_tensor)
            leapp.stop()
            self.fail("Expected an exception for cross-context torch.where")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))
    
    def test_annotate_multiple_parallel_inputs_with_same_name(self):
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            output_tensors = []
            for i in range(10):
                tensor1 = annotate.input_tensors('func_combined', {'input': torch.tensor([1.0, 2.0, 3.0])})
                tensor1 += i
                output_tensors.append(tensor1)

            annotate.output_tensors('func_combined', {'outputs': output_tensors}, export_with="jit")

            leapp.stop()
            leapp.compile_graph(visualize=False)
            self.fail("Expected an exception")

        except Exception as e:
            self.assertIn("Duplicate name ", str(e))
            self.assertIn("Each input/output must have a unique name", str(e))

    def test_output_tensors_with_non_traced_tensor(self):
        """Test that passing non-TracedTensors to output_tensors raises the correct error.
        
        This happens when the user doesn't use the returned TracedTensors from input_tensors()
        in their computations.
        """
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            
            # Get traced tensors but don't use them
            traced_input = annotate.input_tensors('func', {'input': torch.tensor([1.0, 2.0, 3.0])})
            
            # Create a completely new tensor (not derived from traced_input)
            # This is the user error - they should be using traced_input
            untraced_output = torch.tensor([4.0, 5.0, 6.0])
            
            # This should fail because untraced_output is not a TracedTensor
            annotate.output_tensors('func', {'output': untraced_output}, export_with="jit")
            
            leapp.stop()
            self.fail("Expected an exception")
            
        except Exception as e:
            self.assertIn("non-traced tensors", str(e))

    def test_traced_tensor_as_static_output_fails(self):
        """Test that using a TracedTensor (derived from input) as a static output fails.
        
        Static outputs should be constant tensors that are NOT derived from inputs.
        If a user accidentally marks a computed tensor as static, it should error.
        """
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            
            # Create input and trace it
            input_tensor = torch.tensor([1.0, 2.0, 3.0])
            traced_input = annotate.input_tensors('func', {'input': input_tensor})
            
            # Compute a tensor from the traced input
            computed_tensor = traced_input + 1.0  # This is a TracedTensor
            
            # Create a proper output
            proper_output = traced_input * 2.0
            
            # User error: trying to use a TracedTensor as a static output
            # This should fail because static outputs must be raw tensors
            annotate.output_tensors(
                'func',
                {'output': proper_output},
                static_outputs={'bad_static': computed_tensor},  # Error: TracedTensor not allowed
                export_with="jit"
            )
            
            leapp.stop()
            self.fail("Expected an exception when using TracedTensor as static output")
            
        except Exception as e:
            error_msg = str(e)
            # Should mention that static outputs cannot be TracedTensors
            self.assertTrue(
                "output_tensors" in error_msg,
                f"Expected error about output_tensors, got: {error_msg}"
            )

    def test_input_tensors_after_output_tensors_same_node(self):
        """Test that calling input_tensors after output_tensors for the same node fails."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            traced = annotate.input_tensors('func', {'input': torch.tensor([1.0, 2.0, 3.0])})
            result = traced + 1.0
            annotate.output_tensors('func', {'output': result}, export_with="jit")
            annotate.input_tensors('func', {'input2': torch.tensor([4.0, 5.0, 6.0])})
            leapp.stop()
            self.fail("Expected an exception for input_tensors after output_tensors on same node")
        except Exception as e:
            leapp.stop()
            self.assertIn("Cannot reuse a node name from a different call site.", str(e))

    def test_input_tensors_after_method_same_name(self):
        """Test that calling input_tensors with a name already used by a method fails."""
        @annotate.method(export_with="jit")
        def funcA(inputA: torch.Tensor):
            return inputA + 1.0

        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            funcA(torch.tensor([1.0, 2.0, 3.0]))
            annotate.input_tensors('funcA', {'input2': torch.tensor([4.0, 5.0, 6.0])})
            leapp.stop()
            self.fail("Expected an exception for input_tensors reusing a method node name")
        except Exception as e:
            leapp.stop()
            self.assertIn("Cannot reuse a node name from a different call site.", str(e))

    def test_expected_fail_connection_shape_mismatch(self):
        """Shape mismatch should fail at compile_graph edge compatibility validation."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)

            source_input = annotate.input_tensors('source_node', {'x': torch.tensor([1.0, 2.0, 3.0])})
            source_output = source_input + 1.0
            annotate.output_tensors('source_node', {'y': source_output}, export_with="jit")

            # mutate shape after output_tensors and before next input_tensors,
            # while preserving LEAPP tag so connection is still established
            mutated = source_output.clone().reshape(1, 3)
            mutated.leapp_tag = source_output.leapp_tag

            target_input = annotate.input_tensors('target_node', {'y': mutated})
            target_output = target_input + 2.0
            annotate.output_tensors('target_node', {'z': target_output}, export_with="jit")

            leapp.stop()
            leapp.compile_graph(visualize=False)
            self.fail("Expected an exception for shape mismatch")
        except Exception as e:
            try:
                leapp.stop()
            except Exception:
                pass
            self.assertIn("Shape mismatch in pipeline connection", str(e))

    def test_expected_fail_connection_dtype_mismatch(self):
        """Dtype mismatch should fail at compile_graph edge compatibility validation."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)

            source_input = annotate.input_tensors('source_node', {'x': torch.tensor([1.0, 2.0, 3.0])})
            source_output = source_input + 1.0
            annotate.output_tensors('source_node', {'y': source_output}, export_with="jit")

            # mutate dtype after output_tensors and before next input_tensors,
            # while preserving LEAPP tag so connection is still established
            mutated = source_output.clone().to(torch.int32)
            mutated.leapp_tag = source_output.leapp_tag

            target_input = annotate.input_tensors('target_node', {'y': mutated})
            target_output = target_input + 2
            annotate.output_tensors('target_node', {'z': target_output}, export_with="jit")

            leapp.stop()
            leapp.compile_graph(visualize=False)
            self.fail("Expected an exception for dtype mismatch")
        except Exception as e:
            try:
                leapp.stop()
            except Exception:
                pass
            self.assertIn("Dtype mismatch in pipeline connection", str(e))

    def test_expected_fail_feedback_connection_shape_mismatch(self):
        """Feedback edge mismatch should fail during compatibility validation."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)

            obs = torch.tensor([1.0, 2.0, 3.0])
            feedback_next = torch.tensor([0.0, 0.0, 0.0])

            # run twice from identical call sites to establish a feedback edge
            for _ in range(2):
                obs_traced, fb_traced = annotate.input_tensors('node_a', {'obs': obs, 'fb': feedback_next})
                a_out = obs_traced + fb_traced
                annotate.output_tensors('node_a', {'a_out': a_out}, export_with="jit")

                c_in = annotate.input_tensors('node_c', {'a_out': a_out})
                c_out = c_in.reshape(1, 3)  # source output descriptor shape is [1, 3]
                annotate.output_tensors('node_c', {'fb_out': c_out}, export_with="jit")

                # mutate value between output_tensors and next input_tensors
                # preserve tag to create feedback connection into node_a/fb
                feedback_next = c_out.reshape(3).clone()
                feedback_next.leapp_tag = c_out.leapp_tag

            leapp.stop()
            leapp.compile_graph(visualize=False)
            self.fail("Expected an exception for feedback shape mismatch")
        except Exception as e:
            try:
                leapp.stop()
            except Exception:
                pass
            self.assertIn("Shape mismatch in pipeline connection", str(e))

    # -------------------------------------------------------------------
    # Caller identity tests — verify normalized annotation-origin detection
    # -------------------------------------------------------------------

    def test_caller_identity_wrapper_indirection(self):
        """Two different helpers calling input_tensors for the same node
        should be detected as different call sites."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            inp = _helper_path_a('func', torch.tensor([1.0, 2.0, 3.0]))
            result = inp + 1.0
            annotate.output_tensors('func', {'output': result}, export_with="onnx")
            _helper_path_b('func', torch.tensor([4.0, 5.0, 6.0]))
            leapp.stop()
            self.fail("Expected an exception for wrapper indirection")
        except Exception as e:
            leapp.stop()
            self.assertIn("Cannot reuse a node name from a different call site.", str(e))

    def test_caller_identity_same_helper_different_caller(self):
        """A single helper called from two different lines in user code
        should now be accepted because the helper itself is the origin."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        inp = _shared_helper('func', torch.tensor([1.0]))
        result = inp + 1.0
        annotate.output_tensors('func', {'output': result}, export_with="onnx")
        _shared_helper('func', torch.tensor([2.0]))
        leapp.stop()

    def test_caller_identity_same_path_in_loop(self):
        """The same helper called from the same line in a loop should be
        accepted — the stack trace is identical on every iteration."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        for i in range(3):
            inp = _shared_helper('func', torch.tensor([float(i)]))
            result = inp + 1.0
            annotate.output_tensors('func', {'output': result}, export_with="onnx")
        leapp.stop()

    def test_caller_identity_conditional_branching(self):
        """input_tensors called from different if/else branches (different
        lines) should be detected as different call sites."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            if True:
                inp = annotate.input_tensors('func', {'input': torch.tensor([1.0])})
            else:
                inp = annotate.input_tensors('func', {'input': torch.tensor([1.0])})
            result = inp + 1.0
            annotate.output_tensors('func', {'output': result}, export_with="onnx")
            if False:
                annotate.input_tensors('func', {'input': torch.tensor([2.0])})
            else:
                annotate.input_tensors('func', {'input': torch.tensor([2.0])})
            leapp.stop()
            self.fail("Expected an exception for conditional branching")
        except Exception as e:
            leapp.stop()
            self.assertIn("Cannot reuse a node name from a different call site.", str(e))

    def test_caller_identity_functools_partial(self):
        """A functools.partial binding called from two different lines
        should be detected as different call sites."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            input_func = functools.partial(annotate.input_tensors, 'func')
            inp = input_func({'input': torch.tensor([1.0])})
            result = inp + 1.0
            annotate.output_tensors('func', {'output': result}, export_with="onnx")
            input_func({'input': torch.tensor([2.0])})
            leapp.stop()
            self.fail("Expected an exception for functools.partial")
        except Exception as e:
            leapp.stop()
            self.assertIn("Cannot reuse a node name from a different call site.", str(e))

    def test_caller_identity_class_hierarchy(self):
        """A base-class method called through different subclass methods
        should now be accepted because the annotation anchor is shared."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        a = _PipelineA()
        b = _PipelineB()
        inp = a.run(torch.tensor([1.0]))
        result = inp + 1.0
        annotate.output_tensors('func', {'output': result}, export_with="onnx")
        b.run(torch.tensor([2.0]))
        leapp.stop()

    def test_caller_identity_decorator_adds_frame(self):
        """Calling a raw function vs its decorated version should be
        accepted when the decorated path resolves to the same anchor."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        inp = _raw_annotate(torch.tensor([1.0]))
        result = inp + 1.0
        annotate.output_tensors('func', {'output': result}, export_with="onnx")
        _decorated_annotate(torch.tensor([2.0]))
        leapp.stop()

    def test_caller_identity_lambda_vs_named(self):
        """input_tensors called through a named helper vs a lambda should
        be detected as different call sites."""
        try:
            leapp.start(name=self.TEST_GRAPH_NAME)
            inp = _shared_helper('func', torch.tensor([1.0]))
            result = inp + 1.0
            annotate.output_tensors('func', {'output': result}, export_with="onnx")
            def via_lambda(t):
                return annotate.input_tensors('func', {'input': t})
            via_lambda(torch.tensor([2.0]))
            leapp.stop()
            self.fail("Expected an exception for lambda vs named function")
        except Exception as e:
            leapp.stop()
            self.assertIn("Cannot reuse a node name from a different call site.", str(e))

if __name__ == '__main__':
    unittest.main(verbosity=2)
