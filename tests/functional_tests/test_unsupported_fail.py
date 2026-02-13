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


class TestUnsupportedFail(LEAPPFunctionalTestBase):
    """Unit tests to see if unsupported io is properly handled"""

    def test_same_variable_used_twice(self):

        @annotate.method()
        def funcA(inputA: torch.Tensor):
            return inputA

        @annotate.method()
        def funcB(inputB: torch.Tensor, inputC: torch.Tensor):
            return inputB+inputC

        annotate.start(name=self.TEST_GRAPH_NAME)
        retvalA = funcA(torch.tensor([1, 2, 3]))
        funcB(retvalA, retvalA)
        annotate.stop()

        try:
            annotate.compile_graph()
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

        annotate.start(name=self.TEST_GRAPH_NAME)

        try:
            return_value = func(torch.tensor([1, 2, 3]))
            func_copy(return_value)
        except Exception as e:
            annotate.stop()
            first_line = str(e).split('\n')[0]
            expected = "Error: funcA seen twice but block boundaries do not match"
            self.assertEqual(first_line, expected)
            return

        annotate.stop()
        self.fail("Expected an exception")

    def test_io_reconciliation_name_overlap(self):
        @annotate.method()
        def funcA(input: torch.Tensor):
            detections = torch.zeros(input.shape)
            # some processing
            return detections

        @annotate.method()
        def funcB(detections):
            retval = torch.tensor([])
            # some processing
            return retval

        @annotate.method()
        def funcC(input, detections):
            retval = torch.tensor([])
            return retval

        annotate.start(name=self.TEST_GRAPH_NAME)
        detections = funcA(torch.tensor([1.0, 2.0, 3.0]))
        funcB(detections)
        funcC(detections, torch.tensor([1.0, 2.0, 3.0]))
        annotate.stop()
        try:
            annotate.compile_graph()
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        
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
        
        annotate.stop()

    def test_reentrant_tracing_block_inside_method(self):
        """Test that re-entrant tracing is properly rejected.
        
        Attempting to use annotate.block inside a function decorated with
        annotate.method should fail because the tracing lock is already acquired.
        """
        @annotate.method()
        def outer_func(inputA: torch.Tensor):
            # This should fail - trying to start a block while already tracing
            with annotate.block("inner_block"):
                result = inputA * 2.0
            return result

        annotate.start(name=self.TEST_GRAPH_NAME)
        
        try:
            outer_func(torch.tensor([1.0, 2.0, 3.0]))
            annotate.stop()
            self.fail("Expected an exception for re-entrant tracing")
        except Exception as e:
            annotate.stop()
            # The error should mention that tracing is already active
            self.assertIn("attempting to set up new trace", str(e).lower())

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

        annotate.start(name=self.TEST_GRAPH_NAME)
        
        try:
            outer_func(torch.tensor([1.0, 2.0, 3.0]))
            annotate.stop()
            self.fail("Expected an exception for re-entrant tracing")
        except Exception as e:
            annotate.stop()
            # The error should mention that we're trying to set up a new trace
            self.assertIn("attempting to set up new trace", str(e).lower())
    
    def test_reentrant_tracing_using_traced_tensors(self): #TODO: these both need to fail
        try:
            annotate.start(name=self.TEST_GRAPH_NAME)
            tensors = annotate.input_tensors('func', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensors += 100
            input = torch.tensor([1.0, 2.0, 3.0])
            @annotate.method()
            def inner_func(inputA: torch.Tensor):
                return inputA + tensors
            output_tensors = inner_func(input)
            annotate.output_tensors('func', {'outputA': output_tensors}, export_with="jit")
            annotate.stop()
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("Cannot use TracedTensor", str(e))
            self.assertIn("Call output_tensors() first", str(e))
    
    def test_passing_traced_tensor_to_method(self):
        try:
            @annotate.method()
            def inner_func(inputA: torch.Tensor):
                return inputA + 5
            annotate.start(name=self.TEST_GRAPH_NAME)
            tensors = annotate.input_tensors('func', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensors += 100
            output_tensors = inner_func(tensors)
            annotate.output_tensors('func', {'outputA': output_tensors}, export_with="jit")
            annotate.stop()
            self.fail("Expected an exception")
        except Exception as e:
            error_msg = str(e)
            self.assertIn("Cannot use TracedTensor", error_msg)
            self.assertIn("Call output_tensors() first", error_msg)
    
    def test_cross_context_traced_tensor_usage(self):
        """Basic test: addition of two TracedTensors from different contexts."""
        try:
            annotate.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([1.0, 2.0, 3.0])})
            output_tensors = tensor1 + tensor2
            annotate.output_tensors('func1', {'outputA': output_tensors}, export_with="jit")
            annotate.stop()
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))
            self.assertIn("Call output_tensors() to finalize one of the TracedTensor nodes first", str(e))

    def test_cross_context_torch_cat_list(self):
        """Test torch.cat with a list containing TracedTensors from different contexts."""
        try:
            annotate.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([4.0, 5.0, 6.0])})
            # torch.cat takes a list of tensors - should detect cross-context in the list
            output_tensors = torch.cat([tensor1, tensor2], dim=0)
            annotate.stop()
            self.fail("Expected an exception for cross-context torch.cat")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))

    def test_cross_context_torch_stack(self):
        """Test torch.stack with tensors from different contexts."""
        try:
            annotate.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([4.0, 5.0, 6.0])})
            # torch.stack also takes a list
            output_tensors = torch.stack([tensor1, tensor2], dim=0)
            annotate.stop()
            self.fail("Expected an exception for cross-context torch.stack")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))

    def test_cross_context_matmul(self):
        """Test matrix multiplication with TracedTensors from different contexts."""
        try:
            annotate.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([[5.0, 6.0], [7.0, 8.0]])})
            # Matrix multiplication with two explicit args
            output_tensors = torch.matmul(tensor1, tensor2)
            annotate.stop()
            self.fail("Expected an exception for cross-context torch.matmul")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))

    def test_cross_context_torch_add_with_alpha(self):
        """Test torch.add with alpha kwarg - mixing contexts in positional args."""
        try:
            annotate.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([4.0, 5.0, 6.0])})
            # torch.add(input, other, alpha=1) - tests kwargs handling
            output_tensors = torch.add(tensor1, tensor2, alpha=2.0)
            annotate.stop()
            self.fail("Expected an exception for cross-context torch.add with alpha")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))

    def test_cross_context_three_contexts(self):
        """Test detection with three different contexts mixed together."""
        try:
            annotate.start(name=self.TEST_GRAPH_NAME)
            tensor1 = annotate.input_tensors('func1', {'inputA': torch.tensor([1.0, 2.0, 3.0])})
            tensor2 = annotate.input_tensors('func2', {'inputB': torch.tensor([4.0, 5.0, 6.0])})
            tensor3 = annotate.input_tensors('func3', {'inputC': torch.tensor([7.0, 8.0, 9.0])})
            # Mix all three contexts
            output_tensors = torch.cat([tensor1, tensor2, tensor3], dim=0)
            annotate.stop()
            self.fail("Expected an exception for three cross-contexts")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))

    def test_cross_context_torch_where(self):
        """Test torch.where with condition, x, y from different contexts."""
        try:
            annotate.start(name=self.TEST_GRAPH_NAME)
            # condition from one context
            cond_tensor = annotate.input_tensors('cond_ctx', {'cond': torch.tensor([True, False, True])})
            # x from another context
            x_tensor = annotate.input_tensors('x_ctx', {'x': torch.tensor([1.0, 2.0, 3.0])})
            # y is a regular tensor (should be fine)
            y_tensor = torch.tensor([10.0, 20.0, 30.0])
            # torch.where(condition, x, y) - two TracedTensors from different contexts
            output_tensors = torch.where(cond_tensor, x_tensor, y_tensor)
            annotate.stop()
            self.fail("Expected an exception for cross-context torch.where")
        except Exception as e:
            self.assertIn("Cannot mix multiple active TracedTensors from different contexts", str(e))
    
    def test_annotate_multiple_parallel_inputs_with_same_name(self):
        try:
            annotate.start(name=self.TEST_GRAPH_NAME)
            output_tensors = []
            for i in range(10):
                tensor1 = annotate.input_tensors('func_combined', {'input': torch.tensor([1.0, 2.0, 3.0])})
                tensor1 += i
                output_tensors.append(tensor1)

            annotate.output_tensors('func_combined', {'outputs': output_tensors}, export_with="jit")

            annotate.stop()
            annotate.compile_graph(visualize=False)
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
            annotate.start(name=self.TEST_GRAPH_NAME)
            
            # Get traced tensors but don't use them
            traced_input = annotate.input_tensors('func', {'input': torch.tensor([1.0, 2.0, 3.0])})
            
            # Create a completely new tensor (not derived from traced_input)
            # This is the user error - they should be using traced_input
            untraced_output = torch.tensor([4.0, 5.0, 6.0])
            
            # This should fail because untraced_output is not a TracedTensor
            annotate.output_tensors('func', {'output': untraced_output}, export_with="jit")
            
            annotate.stop()
            self.fail("Expected an exception")
            
        except Exception as e:
            self.assertIn("output_tensors declaration", str(e))

    def test_traced_tensor_as_static_output_fails(self):
        """Test that using a TracedTensor (derived from input) as a static output fails.
        
        Static outputs should be constant tensors that are NOT derived from inputs.
        If a user accidentally marks a computed tensor as static, it should error.
        """
        try:
            annotate.start(name=self.TEST_GRAPH_NAME)
            
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
            
            annotate.stop()
            self.fail("Expected an exception when using TracedTensor as static output")
            
        except Exception as e:
            error_msg = str(e)
            # Should mention that static outputs cannot be TracedTensors
            self.assertTrue(
                "output_tensors declaration" in error_msg,
                f"Expected error about output_tensors declaration, got: {error_msg}"
            )

if __name__ == '__main__':
    unittest.main(verbosity=2)
