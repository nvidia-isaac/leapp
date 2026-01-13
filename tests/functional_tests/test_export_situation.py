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
from .base import LEAPPFunctionalTestBase
import torch
from leapp import annotate


class TestExportSituation(LEAPPFunctionalTestBase):
    """
    Unit tests to see if export situation is properly handled

    These tests test for things that are put inside of the code
    snippet that we want to support

    """

    def test_export_nnModule_function(self):
        linear = torch.nn.Linear(3, 3)

        @annotate.method(export_with="torch", environment_constants=['linear'])
        def funcA(inputA: torch.Tensor):
            output = linear(inputA)
            return output

        annotate.start(name=self.TEST_GRAPH_NAME)
        funcA(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        annotate.stop()
        annotate.compile_graph(visualize=False)

    def test_export_nnModule(self):
        class moduleA(torch.nn.Module):
            def __init__(self):
                super(moduleA, self).__init__()
                self.linear = torch.nn.Linear(3, 3)

            @annotate.method(export_with="torch")
            def forward(self, inputA: torch.Tensor):
                return self.linear(inputA)

        annotate.start(name=self.TEST_GRAPH_NAME)
        moduleA()(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        annotate.stop()
        annotate.compile_graph(visualize=False)

    def test_export_nnModule_with_dict_list_io(self):

        @annotate.method(export_with="torch")
        def funcA(inputA: dict):
            list_conversion = list(inputA.values())
            return list_conversion

        input = {'a': torch.tensor([1]), 'b': torch.tensor(
            [2]), 'c': torch.tensor([3]), 'd': torch.tensor([4])}
        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = funcA(input)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(annotate, nodes=1, inputs=4, outputs=4,
                                    internal_connections=0)
        self.verify_single_torchscript_model_expected_value(
            [input], [expected_output], funcA.__name__)

        model_info = self.inspect_torchscript_model(funcA.__name__)
        # self + 4 flat tensors (inputA_a, inputA_b, inputA_c, inputA_d)
        self.assertEqual(len(model_info['inputs']), 5)
        self.assertEqual(len(model_info['outputs']), 4)  # list_conversion

    def test_export_nnModule_with_large_nested_dict_io(self):
        @annotate.method(export_with="torch")
        def funcA(inputA: dict, inputB):
            underlying_values = inputA[0]['nested']
            list_conversion = list(underlying_values.values())
            return_values = list_conversion + [inputB]
            return {'return_values': [return_values]}

        input = [[{'nested': {'a': torch.tensor([1]), 'b': torch.tensor(
            [2]), 'c': torch.tensor([3]), 'd': torch.tensor([4])}}], torch.tensor([5])]
        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = funcA(*input)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(annotate, nodes=1, inputs=5, outputs=5,
                                    internal_connections=0)
        self.verify_single_torchscript_model_expected_value(
            input, [expected_output], funcA.__name__)

        model_info = self.inspect_torchscript_model(funcA.__name__)
        # self + 5 flat tensors (inputA contains 4 nested tensors + inputB is 1 tensor)
        self.assertEqual(len(model_info['inputs']), 6)
        self.assertEqual(len(model_info['outputs']), 5)  # return_values

    def test_export_class_method_that_relies_on_dynamic_variable(self):
        class moduleA():
            def __init__(self):
                self.idx = 0
                self.stride = 3

            @annotate.method(export_with="torch", environment_constants=['self.idx', 'self.stride'])
            def get_subset(self, inputA: torch.Tensor):
                retval = inputA[self.idx:self.idx+self.stride]
                self.idx += self.stride
                return retval
        moduleA = moduleA()
        annotate.start(name=self.TEST_GRAPH_NAME)
        subset = moduleA.get_subset(torch.tensor([1, 2, 3]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(annotate, nodes=1, inputs=1, outputs=1,
                                    internal_connections=0)
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1, 2, 3])], [subset], moduleA.get_subset.__name__)

    def test_export_block_with_class_variables_as_inputs(self):
        class moduleA():
            def __init__(self):
                self.var1 = None
                self.var2 = None

            def concatenate_vars(self):
                with annotate.block(node_name="concatenate", export_with="torch",
                                    inputs=["self.var1", "self.var2"], outputs=["result"]):
                    result = torch.cat((self.var1, self.var2), dim=-1)
                return result

            def update_vars(self, var1, var2):
                self.var1 = var1
                self.var2 = var2

        input1 = torch.tensor([1])
        input2 = torch.tensor([2])
        module = moduleA()
        annotate.start(name=self.TEST_GRAPH_NAME)
        module.update_vars(input1, input2)
        output = module.concatenate_vars()
        annotate.stop()
        annotate.compile_graph()

        self.verify_single_torchscript_model_expected_value(
            [input1, input2], [output], "concatenate")

    def test_export_block_with_complex_class_variables_as_inputs(self):
        class moduleA():
            def __init__(self):
                self.var1 = None
                self.var2 = None

            def concatenate_vars(self):
                with annotate.block(node_name="concatenate", export_with="torch",
                                    inputs=["self.var1", "self.var2"], outputs=["result"]):
                    # Stack list of tensors before summing (TorchScript compatible)
                    value1 = torch.sum(torch.stack(self.var1))
                    # Convert dict values to list, then stack (TorchScript compatible)
                    value2 = torch.sum(torch.stack(list(self.var2.values())))
                    result = torch.stack([value1, value2])
                return result

            def update_vars(self, var1, var2):
                self.var1 = var1
                self.var2 = var2

        input1 = [torch.tensor([1]), torch.tensor([2])]
        input2 = {'a': torch.tensor([2]), 'b': torch.tensor([3])}
        module = moduleA()
        annotate.start(name=self.TEST_GRAPH_NAME)
        module.update_vars(input1, input2)
        output = module.concatenate_vars()
        annotate.stop()
        annotate.compile_graph()

        self.verify_single_torchscript_model_expected_value(
            [input1, input2], [output], "concatenate")

    def test_export_dict_and_list_bidirectional_io(self):
        """Test function that takes dict and list inputs, returns list and dict outputs"""
        @annotate.method(export_with="torch")
        def test_complex_io(input: dict, input_2: list):
            dictionary_output = {}
            for idx, value in enumerate(input_2):
                dictionary_output[str(idx)] = value
            return [value for value in input.values()], dictionary_output

        input_dict = {
            'a': torch.tensor([1]),
            'b': torch.tensor([2]),
            'c': torch.tensor([3]),
            'd': torch.tensor([4]),
        }
        input_list = [torch.tensor([5]), torch.tensor(
            [6]), torch.tensor([7]), torch.tensor([8])]

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = test_complex_io(input_dict, input_list)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # Verify graph statistics: 1 node, 8 dangling inputs, 8 dangling outputs, 0 internal connections
        self.verify_num_connections(annotate, nodes=1, inputs=8, outputs=8,
                                    internal_connections=0)

        # Verify the model produces the correct output
        self.verify_single_torchscript_model_expected_value(
            [input_dict, input_list], [expected_output], test_complex_io.__name__)

        # Verify model structure: forward should have 8 flat tensor inputs + self
        model_info = self.inspect_torchscript_model(test_complex_io.__name__)
        # self + 4 dict tensors + 4 list tensors
        self.assertEqual(len(model_info['inputs']), 9)
        # list output + dict output
        self.assertEqual(len(model_info['outputs']), 8)

        # Verify output structure
        output_list, output_dict = expected_output
        self.assertEqual(len(output_list), 4)
        self.assertEqual(len(output_dict), 4)
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in output_list))
        self.assertTrue(all(isinstance(v, torch.Tensor)
                        for v in output_dict.values()))

    def test_export_split_tensor_to_list(self):
        """Test function that takes a single tensor and splits it into a list of tensors"""
        @annotate.method(export_with="torch", verbose=True)
        def split_tensor(input_tensor: torch.Tensor):
            # Split the tensor into individual elements
            return [input_tensor[i:i+1] for i in range(len(input_tensor))]

        # Create input tensor with shape (20,) with values 0, 10, 20, ..., 190
        input_tensor = torch.arange(0, 200, 10)

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = split_tensor(input_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # Verify graph statistics: 1 node, 1 dangling input, 20 dangling outputs, 0 internal connections
        self.verify_num_connections(annotate, nodes=1, inputs=1, outputs=20,
                                    internal_connections=0)

        # Verify the model produces the correct output
        self.verify_single_torchscript_model_expected_value(
            [input_tensor], [expected_output], split_tensor.__name__)

        # Verify model structure: forward should have 1 tensor input + self
        model_info = self.inspect_torchscript_model(split_tensor.__name__)
        self.assertEqual(len(model_info['inputs']), 2)  # self + 1 input tensor
        # 20 individual tensor outputs
        self.assertEqual(len(model_info['outputs']), 20)

        # Verify output structure: should be a list of 20 tensors, each with shape (1,)
        self.assertEqual(len(expected_output), 20)
        self.assertTrue(all(isinstance(t, torch.Tensor)
                        for t in expected_output))
        self.assertTrue(
            all(t.shape == torch.Size([1]) for t in expected_output))

        # Verify output values
        for i, tensor in enumerate(expected_output):
            self.assertEqual(tensor.item(), i * 10)

    def test_export_nested_list_of_lists_output(self):
        """Test function that returns a nested list of lists of tensors"""
        @annotate.method(export_with="torch")
        def create_nested_lists(input_tensor: torch.Tensor):
            # Split into groups of 2
            return [[input_tensor[i:i+1], input_tensor[i+1:i+2]] for i in range(0, 4, 2)]

        input_tensor = torch.arange(4)

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = create_nested_lists(input_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # Should have 1 input, 4 outputs (flattened from nested structure)
        self.verify_num_connections(annotate, nodes=1, inputs=1, outputs=4,
                                    internal_connections=0)

        self.verify_single_torchscript_model_expected_value(
            [input_tensor], [expected_output], create_nested_lists.__name__)

        # Verify structure: 2 lists, each containing 2 tensors
        self.assertEqual(len(expected_output), 2)
        self.assertTrue(all(len(sublist) == 2 for sublist in expected_output))

        model_info = self.inspect_torchscript_model(
            create_nested_lists.__name__)
        self.assertEqual(len(model_info['inputs']), 2)  # self + 1 input
        self.assertEqual(len(model_info['outputs']), 4)  # 4 flattened outputs

    def test_export_dict_with_list_values_output(self):
        """Test function that returns a dict where values are lists of tensors"""
        @annotate.method(export_with="torch")
        def create_dict_of_lists(input_list: list):
            return {
                'first_half': [input_list[0], input_list[1]],
                'second_half': [input_list[2], input_list[3]]
            }

        input_list = [torch.tensor([i]) for i in range(4)]

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = create_dict_of_lists(input_list)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # 4 inputs, 4 outputs (dict values flattened)
        self.verify_num_connections(annotate, nodes=1, inputs=4, outputs=4,
                                    internal_connections=0)

        self.verify_single_torchscript_model_expected_value(
            [input_list], [expected_output], create_dict_of_lists.__name__)

        # Verify structure
        self.assertEqual(len(expected_output['first_half']), 2)
        self.assertEqual(len(expected_output['second_half']), 2)

        model_info = self.inspect_torchscript_model(
            create_dict_of_lists.__name__)
        self.assertEqual(len(model_info['inputs']), 5)  # self + 4 inputs
        self.assertEqual(len(model_info['outputs']), 4)  # 4 flattened outputs

    def test_export_mixed_return_types(self):
        """Test function that returns a single tensor, a list, and a dict all together"""
        @annotate.method(export_with="torch")
        def mixed_outputs(input_tensor: torch.Tensor):
            single_tensor = input_tensor[0:1]  # Still a tensor, shape (1,)
            list_out = [input_tensor[1:2], input_tensor[2:3]]
            dict_out = {'a': input_tensor[3:4], 'b': input_tensor[4:5]}
            return single_tensor, list_out, dict_out

        input_tensor = torch.arange(5)

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = mixed_outputs(input_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # 1 input, 5 outputs (1 tensor + 2 list tensors + 2 dict tensors)
        self.verify_num_connections(annotate, nodes=1, inputs=1, outputs=5,
                                    internal_connections=0)

        self.verify_single_torchscript_model_expected_value(
            [input_tensor], [expected_output], mixed_outputs.__name__)

        # Verify structure: all outputs are tensors
        single_tensor, list_out, dict_out = expected_output
        self.assertIsInstance(single_tensor, torch.Tensor)
        self.assertEqual(len(list_out), 2)
        self.assertTrue(all(isinstance(t, torch.Tensor) for t in list_out))
        self.assertEqual(len(dict_out), 2)
        self.assertTrue(all(isinstance(t, torch.Tensor)
                        for t in dict_out.values()))

        model_info = self.inspect_torchscript_model(mixed_outputs.__name__)
        self.assertEqual(len(model_info['inputs']), 2)  # self + 1 input
        self.assertEqual(len(model_info['outputs']), 5)  # 5 flattened outputs

    def test_export_list_of_dicts_output(self):
        """Test function that returns a list of dictionaries"""
        @annotate.method(export_with="torch")
        def create_list_of_dicts(input_tensor: torch.Tensor):
            return [
                {'x': input_tensor[0:1], 'y': input_tensor[1:2]},
                {'x': input_tensor[2:3], 'y': input_tensor[3:4]}
            ]

        input_tensor = torch.arange(4)

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = create_list_of_dicts(input_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # 1 input, 4 outputs (flattened from list of dicts)
        self.verify_num_connections(annotate, nodes=1, inputs=1, outputs=4,
                                    internal_connections=0)

        self.verify_single_torchscript_model_expected_value(
            [input_tensor], [expected_output], create_list_of_dicts.__name__)

        # Verify structure: list of 2 dicts, each with 2 keys
        self.assertEqual(len(expected_output), 2)
        self.assertTrue(all(len(d) == 2 for d in expected_output))
        self.assertTrue(all('x' in d and 'y' in d for d in expected_output))

        model_info = self.inspect_torchscript_model(
            create_list_of_dicts.__name__)
        self.assertEqual(len(model_info['inputs']), 2)  # self + 1 input
        self.assertEqual(len(model_info['outputs']), 4)  # 4 flattened outputs

    def test_export_deeply_nested_dict_to_flat_list(self):
        """Test function with deeply nested dict input and flat list output"""
        @annotate.method(export_with="torch")
        def flatten_nested_structure(nested: dict):
            # Extract from deeply nested structure: nested['level1']['level2']['data']
            level2 = nested['level1']['level2']
            return [level2['data'][key] for key in sorted(level2['data'].keys())]

        nested_input = {
            'level1': {
                'level2': {
                    'data': {
                        'a': torch.tensor([1]),
                        'b': torch.tensor([2]),
                        'c': torch.tensor([3])
                    }
                }
            }
        }

        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = flatten_nested_structure(nested_input)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # 3 inputs (flattened from nested dict), 3 outputs
        self.verify_num_connections(annotate, nodes=1, inputs=3, outputs=3,
                                    internal_connections=0)

        self.verify_single_torchscript_model_expected_value(
            [nested_input], [expected_output], flatten_nested_structure.__name__)

        # Verify output is a list of 3 tensors
        self.assertEqual(len(expected_output), 3)
        self.assertTrue(all(isinstance(t, torch.Tensor)
                        for t in expected_output))

        model_info = self.inspect_torchscript_model(
            flatten_nested_structure.__name__)
        # self + 3 nested inputs
        self.assertEqual(len(model_info['inputs']), 4)
        self.assertEqual(len(model_info['outputs']), 3)  # 3 outputs

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_export_buffer_capture_across_iterations(self):
        """
        Test that buffers are captured from the first iteration only.

        This test creates:
        - A method with a counter buffer that increments INSIDE the function
        - A block with another counter that increments INSIDE the block

        During Python execution, buffers change each iteration.
        The exported model should freeze buffer values from the first iteration,
        so inference always produces the same output as the first iteration.

        Example: count_multiple(value) increments count then returns value * count
        - Python iter 1: count 1->2, returns value*2
        - Python iter 2: count 2->3, returns value*3
        - Exported model: frozen at count=2, always returns value*2
        """
        from leapp.inference_manager import InferenceManager

        device = torch.device('cuda')

        class CounterModel:
            def __init__(self):
                # Counters that will be incremented inside functions (on CUDA)
                self.method_count = torch.tensor([0.0], device=device)
                self.block_count = torch.tensor([10.0], device=device)

            @annotate.method(export_with="torch", register_buffers=['self.method_count'])
            def count_multiply(self, value: torch.Tensor):
                # Increment counter INSIDE the function, then use it
                self.method_count = self.method_count + 1.0
                result = value * self.method_count
                return result

            def count_add(self, value: torch.Tensor):
                with annotate.block(
                    node_name="block_counter",
                    export_with="torch",
                    inputs=["value"],
                    outputs=["result"],
                    register_buffers=["self.block_count"]
                ):
                    # Increment counter INSIDE the block, then use it
                    self.block_count = self.block_count - 2.0
                    result = value + self.block_count
                return result

            def run_pipeline(self, input_value: torch.Tensor):
                # First: multiply by method counter
                intermediate = self.count_multiply(input_value)
                # Then: add block counter
                output = self.count_add(intermediate)
                return output

        model = CounterModel()
        input_value = torch.tensor([3.0], device=device)

        # Run for 5 iterations and collect Python outputs
        annotate.start(name=self.TEST_GRAPH_NAME)
        python_outputs = []
        for i in range(5):
            output = model.run_pipeline(input_value)
            python_outputs.append(output.clone())
        annotate.stop()
        annotate.compile_graph(visualize=False)

        model = InferenceManager(f'{self.TEST_GRAPH_NAME}/{self.TEST_GRAPH_NAME}.yaml')
        inputs = {
            'count_multiply/value': input_value
        }

        inference_outputs = []
        for i in range(5):
            outputs = model(inputs)
            inference_outputs.append(outputs['block_counter/result'].clone())

        for i, (actual, expected) in enumerate(zip(python_outputs, inference_outputs)):
            self.assertTrue(
                torch.allclose(actual, expected),
                f"Iteration {i}: got {actual}, expected {expected}"
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
