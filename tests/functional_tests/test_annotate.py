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
from leapp.leapp_graph.traced_tensor import TracedTensor


class TestAnnotateMethod(LEAPPFunctionalTestBase):
    def test_annotate_method(self):
        """tests the basic situation of using the annotate.method decorator"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA

        @annotate.method(export_with="torch")
        def funcC(inputB: torch.Tensor):
            return inputB+5.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        for i in range(10):
            outputA = funcA(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
            outputC = funcC(outputA)
        annotate.stop()

        self.assertEqual(len(annotate.nodes), 2)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 1)
        self.assertEqual(len(annotate.nodes[funcC.__name__].inputs), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].outputs), 1)
        self.assertEqual(len(annotate.nodes[funcC.__name__].outputs), 1)
        self.assertEqual(
            annotate.nodes[funcA.__name__].inputs[0].name, "inputA")
        self.assertEqual(
            annotate.nodes[funcC.__name__].inputs[0].name, "inputB")

    def test_annotate_method_with_kwargs_and_default_value(self):
        """tests the situation where the function has and default values"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)):
            return inputA

        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA = funcA()
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 0)

        expected_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        # check if using default
        self.verify_single_torchscript_model_expected_value(
            [], [expected_output], funcA.__name__)
        # Or get structured data
        model_info = self.inspect_torchscript_model(funcA.__name__)
        self.assertEqual(len(model_info['inputs']), 1)
        self.assertEqual(len(model_info['outputs']), 1)

    def test_annotate_method_ignoring_default_values(self):
        """tests the situation where we pass in a value overriding the default"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)):
            return inputA

        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA = funcA(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 1)
        model_info = self.inspect_torchscript_model(funcA.__name__)
        self.assertEqual(len(model_info['inputs']), 2)
        self.assertEqual(len(model_info['outputs']), 1)

    def test_annotate_method_ignoring_middle_kwargs(self):
        """tests the situation where the user provides kwargs out of order"""
        default_tensor = torch.tensor([0])

        @annotate.method(export_with="torch")
        def funcA(input1=default_tensor, input2=default_tensor, input3=default_tensor,
                  input4=default_tensor, input5=default_tensor):
            output = torch.cat([input1, input2, input3, input4, input5], dim=0)
            return output

        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA = funcA(input1=torch.tensor([1]), input4=torch.tensor([1]))
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 2)
        model_info = self.inspect_torchscript_model(funcA.__name__)
        self.assertEqual(len(model_info['inputs']), 3)
        self.assertEqual(len(model_info['outputs']), 1)

        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1]), torch.tensor([1])], [outputA], funcA.__name__)

    def test_annotate_method_kwargs_out_of_order(self):
        """tests the situation where the user provides kwargs out of order"""
        default_tensor = torch.tensor([0])

        @annotate.method(export_with="torch")
        def funcA(input1=default_tensor, input2=default_tensor, input3=default_tensor,
                  input4=default_tensor, input5=default_tensor):
            output = torch.cat([input1, input2, input3, input4, input5], dim=0)
            return output
        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA = funcA(input4=torch.tensor(
            [2]), input1=torch.tensor([1]), input5=torch.tensor([3]))
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 3)
        input_format = [input['name'] for input in annotate.detected_nodes[funcA.__name__]['inputs']]
        self.assertEqual(input_format, ['input1', 'input4', 'input5'])
        model_info = self.inspect_torchscript_model(funcA.__name__)
        self.assertEqual(len(model_info['inputs']), 4)
        self.assertEqual(len(model_info['outputs']), 1)
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])], [outputA], funcA.__name__)

    def test_annotate_method_with_multiple_unnamed_returns(self):
        """tests the situation where the function has multiple unnamed returns"""
        @annotate.method(export_with="torch")
        def funcA(input1: torch.Tensor):
            return input1+1, input1+2, input1+3

        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA, outputB, outputC = funcA(torch.tensor([1]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].outputs), 3)

        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1])], [outputA, outputB, outputC], funcA.__name__)

    def test_annotate_method_with_custom_returns(self):
        """tests the situation where the function has custom returns"""

        class MockModule:
            def __init__(self):
                self.counter = torch.tensor([0])

            @annotate.method(node_name="counter", export_with="torch", outputs=["self.counter"], register_buffers=["self.counter"])
            def count(self):
                self.counter += 1

        mock_module = MockModule()

        annotate.start(name=self.TEST_GRAPH_NAME)
        mock_module.count()
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes['counter'].inputs), 0)
        self.assertEqual(len(annotate.nodes['counter'].outputs), 1)
        self.assertEqual(
            annotate.detected_nodes['counter']['outputs'][0]['type'], 'tensor')
        self.verify_single_torchscript_model_expected_value(
            [], [torch.tensor([1])], 'counter')

    def test_annotate_method_with_mixed_returns(self):
        """tests the situation where the function has both default and custom returns"""

        class MockModule:
            def __init__(self):
                self.counter = torch.tensor([0])

            @annotate.method(node_name="counter", export_with="torch", outputs=["self.counter"], register_buffers=["self.counter"])
            def count(self, input: torch.Tensor):
                self.counter += 1
                retval = input*self.counter
                return retval

        mock_module = MockModule()

        annotate.start(name=self.TEST_GRAPH_NAME)
        output = mock_module.count(torch.tensor([1]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes['counter'].inputs), 1)
        self.assertEqual(len(annotate.nodes['counter'].outputs), 2)
        self.assertEqual(
            annotate.detected_nodes['counter']['outputs'][0]['type'], 'tensor')
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1])], [output, torch.tensor([1])], 'counter')

    def test_annotate_method_with_mixed_returns_in_multiple_locations(self):
        """tests the situation where the function has both default and custom returns in multiple locations"""
        class MockModule:
            def __init__(self):
                self.counter = torch.tensor([0])

            @annotate.method(node_name="counter", export_with="torch", outputs=["self.counter"], register_buffers=["self.counter"])
            def count(self, input: torch.Tensor):
                self.counter += 1
                if input.sum() > 0:
                    return input*self.counter
                else:
                    retval = input+self.counter
                    return retval

        mock_module = MockModule()

        annotate.start(name=self.TEST_GRAPH_NAME)
        output = mock_module.count(torch.tensor([1]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes['counter'].inputs), 1)
        self.assertEqual(len(annotate.nodes['counter'].outputs), 2)
        self.assertEqual(
            annotate.detected_nodes['counter']['outputs'][0]['type'], 'tensor')
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([2])], [torch.tensor([2]), torch.tensor([1])], 'counter')

        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([0])], [torch.tensor([1]), torch.tensor([1])], 'counter')



class TestAnnotateTensor(LEAPPFunctionalTestBase):

    def test_annotate_tensor_single(self):
        annotate.start(name=self.TEST_GRAPH_NAME, verbose=True)
        tensor1 = annotate.input_tensors(
            {'input1': torch.tensor([1.0, 2.0, 3.0])}, 'func1')
        tensor1 = tensor1 + 1.0
        tensor1 = tensor1 - 2.0
        tensor1 = tensor1 * 3.0
        tensor1 = tensor1 / 4.0
        tensor1 = tensor1.matmul(torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        annotate.output_tensors(
            {'output1': tensor1}, 'func1', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0)
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1.0, 2.0, 3.0])], [tensor1.tensor], 'func1')

    def test_annotate_tensor_with_dict_io(self):
        annotate.start(name=self.TEST_GRAPH_NAME, verbose=True)
        input_dict = {'input1': torch.tensor([1.0, 2.0, 3.0])}
        input_dict = annotate.input_tensors(
            {'input_dict': input_dict}, 'func1')
        input_dict['input1'] = input_dict['input1'].matmul(torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        annotate.output_tensors(input_dict, 'func1', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0)

    def test_annotate_tensor_with_list_io(self):
        annotate.start(name=self.TEST_GRAPH_NAME, verbose=True)
        tensor_list = [torch.tensor([1.0, 2.0, 3.0]),
                       torch.tensor([4.0, 5.0, 6.0])]
        tensor_list = annotate.input_tensors(
            {'tensor_list': tensor_list}, 'func1')
        tensor_list[0] = tensor_list[0].matmul(torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        tensor_list[1] = tensor_list[1].matmul(torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        annotate.output_tensors(tensor_list, 'func1', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate, nodes=1, inputs=2, outputs=2, internal_connections=0)

    def test_annotate_tensor_with_list_input_dict_output(self):
        """Test with list input and dict output."""
        annotate.start(name=self.TEST_GRAPH_NAME, verbose=True)
        tensor_list = [torch.tensor([1.0, 2.0, 3.0]),
                       torch.tensor([4.0, 5.0, 6.0])]
        tensor_list = annotate.input_tensors(
            {'tensor_list': tensor_list}, 'func1')
        # Process tensors
        processed_0 = tensor_list[0] * 2.0
        processed_1 = tensor_list[1] * 3.0
        # Output as dict
        output_dict = {'result_a': processed_0, 'result_b': processed_1}
        annotate.output_tensors(output_dict, 'func1', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate, nodes=1, inputs=2, outputs=2, internal_connections=0)

    def test_annotate_tensor_with_list_and_single_input_concat_output(self):
        """Test with 2 inputs: one list of 2 tensors, one regular tensor.
        Output is all tensors concatenated together."""
        annotate.start(name=self.TEST_GRAPH_NAME, verbose=True)
        # First input: a list of 2 tensors
        tensor_list = [torch.tensor([1.0, 2.0, 3.0]),
                       torch.tensor([4.0, 5.0, 6.0])]
        tensor_list = annotate.input_tensors(
            {'tensor_list': tensor_list}, 'func1')
        # Second input: a single tensor
        single_tensor = torch.tensor([7.0, 8.0, 9.0])
        single_tensor = annotate.input_tensors(
            {'single_tensor': single_tensor}, 'func1')
        # Concatenate all tensors together
        concatenated = torch.cat([tensor_list[0], tensor_list[1], single_tensor], dim=0)
        # Output as single tensor
        annotate.output_tensors({'concatenated': concatenated}, 'func1', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate, nodes=1, inputs=3, outputs=1, internal_connections=0)

    def test_annotate_tensor_sequential_context(self):
        annotate.start(name=self.TEST_GRAPH_NAME, verbose=True)
        tensor1 = annotate.input_tensors(
            {'input1': torch.tensor([1.0, 2.0, 3.0])}, 'func1')
        tensor2 = tensor1 + 1.0
        annotate.output_tensors({'output1': tensor2}, 'func1', export_with="torch")
        tensor2 = annotate.input_tensors({'input2': tensor2}, 'func2')
        tensor3 = tensor2 + 2.0
        annotate.output_tensors({'output2': tensor3}, 'func2', export_with="torch")
        tensor3 = annotate.input_tensors({'input3': tensor3}, 'func3')
        tensor4 = tensor3 + 3.0
        annotate.output_tensors({'output3': tensor4}, 'func3', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=3, inputs=1, outputs=1, internal_connections=2)

    def test_annotate_multiple_parallel_contexts(self):
        
        annotate.start(name=self.TEST_GRAPH_NAME)
        for i in range(10):
            tensor1 = annotate.input_tensors({'input1': torch.tensor([1.0, 2.0, 3.0])}, f'func{i}')
            tensor1 += i
            annotate.output_tensors({'output1': tensor1}, f'func{i}', export_with="torch")

        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=10, inputs=10, outputs=10, internal_connections=0)
        self.verify_all_models_exist('func0', 'func1', 'func2', 'func3', 'func4', 'func5', 'func6', 'func7', 'func8', 'func9')
    
    def test_annotate_multiple_parallel_inputs_as_one_context(self):
        
        annotate.start(name=self.TEST_GRAPH_NAME)
        output_tensors = []
        for i in range(10):
            tensor1 = annotate.input_tensors({f'input{i}': torch.tensor([1.0, 2.0, 3.0])}, 'func_combined')
            tensor1 += i
            output_tensors.append(tensor1)

        annotate.output_tensors({'outputs': output_tensors}, 'func_combined', export_with="torch")

        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=10, outputs=10, internal_connections=0)
        self.verify_all_models_exist('func_combined')
    
    def test_annotate_node_with_trimmed_inputs(self):
        '''test the situation where the node has inputs that are not used in the computation'''
        annotate.start(name=self.TEST_GRAPH_NAME)
        for i in range(10):
            tensor1 = annotate.input_tensors({'input1': torch.tensor([1.0, 2.0, 3.0])}, 'func1')
            tensor2 = annotate.input_tensors({'input2': torch.tensor([4.0, 5.0, 6.0])}, 'func1')
            tensor3 = tensor1 + 20
            annotate.output_tensors({'output1': tensor3}, 'func1', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        result = self.inspect_torchscript_model('func1')
        self.assertTrue('input2' not in result['inputs'], "input2 should be trimmed from the model")
        self.assertTrue(len(result['inputs']) == 2, "Unexpected number of inputs in the model")
        self.assertTrue(len(result['outputs']) == 1, "Unexpected number of outputs in the model")

        self.verify_all_models_exist('func1')
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0)
    
    def test_annotate_node_with_many_trimmed_inputs(self):
        annotate.start(name=self.TEST_GRAPH_NAME)
        for i in range(5):
            inputs = [torch.tensor([float(i), float(i+1), float(i+2)], dtype=torch.float32) for i in range(20)]
            inputs = annotate.input_tensors({'inputs': inputs}, 'func1')
            
            # Start with first traced input, then accumulate (keeps output in traced graph)
            output = torch.zeros(3, dtype=torch.float32)
            for i in range(0, 15):
                output = output + inputs[i]
            
            annotate.output_tensors({'output': output}, 'func1', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        result = self.inspect_torchscript_model('func1')
        self.assertEqual(len(result['inputs']), 16, "Unexpected number of inputs in the model")
        self.assertEqual(len(result['outputs']), 1, "Unexpected number of outputs in the model")

        self.verify_all_models_exist('func1')
        # 15 inputs used (inputs 0-14), 5 trimmed (inputs 15-19)
        self.verify_num_connections(
            annotate, nodes=1, inputs=15, outputs=1, internal_connections=0)
        
        inputs = [torch.tensor([float(i), float(i+1), float(i+2)], dtype=torch.float32) for i in range(15)]
        self.verify_single_torchscript_model_expected_value(
            [inputs], [output], 'func1')
    
    def test_annotate_multiple_runs(self):
        annotate.start(name=self.TEST_GRAPH_NAME)


        for i in range(10):
            tensor1 = annotate.input_tensors({'input1': torch.tensor([1.0, 2.0, 3.0])}, 'func1')
            if i == 0:
                self.assertTrue(type(tensor1) is TracedTensor)
            if i > 0:
                self.assertTrue(type(tensor1) is torch.Tensor)
            tensor1 += i
            annotate.output_tensors({'output1': tensor1}, 'func1', export_with="torch")

        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0)
        self.verify_all_models_exist('func1')
    

    def test_annotate_traced_tensors_with_feedback(self):
        '''two traced tensor nodes.
        the first one has 2 inputs 1 output which feeds into the second one. 
        the output of the second one going into one of the inputs of the first one'''
        
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        # Initialize feedback tensor (will be updated each iteration)
        feedback_tensor = torch.tensor([0.0, 0.0, 0.0])
        
        for i in range(20):
            # func1: 2 inputs (input1 and feedback from func2), 1 output
            input1, feedback_tensor = annotate.input_tensors({
                'input1': torch.tensor([1.0, 2.0, 3.0]),
                'feedback_in': feedback_tensor
            }, 'func1')
            output1 = input1 + feedback_tensor
            annotate.output_tensors({'output1': output1}, 'func1', export_with="torch")
            
            # func2: takes output from func1, produces output that feeds back to func1
            input2 = annotate.input_tensors({'input2': output1}, 'func2')
            output2 = input2 * 2.0
            annotate.output_tensors({'output2': output2}, 'func2', export_with="torch")
            
            # Update feedback tensor for next iteration
            feedback_tensor = output2
        
        annotate.stop()
        annotate.compile_graph(visualize=False)
        
        # 2 nodes, 1 external input (input1), 1 external output (output2)
        # 1 internal connection (func1 -> func2)
        # 1 feedback connection (func2 -> func1)
        self.verify_num_connections(
            annotate, nodes=2, inputs=1, outputs=0, 
            internal_connections=1, feedback_connections=1)
        self.verify_all_models_exist('func1', 'func2')

    def test_annotate_traced_tensors_three_node_chain_with_feedback(self):
        '''Three node chain where the last node feeds back to the first.
        func1 -> func2 -> func3 -> (feedback to func1)'''
        
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        feedback_tensor = torch.tensor([0.0, 0.0, 0.0])
        
        for i in range(10):
            # func1: external input + feedback from func3
            input1, fb = annotate.input_tensors({
                'input1': torch.tensor([1.0, 2.0, 3.0]),
                'feedback': feedback_tensor
            }, 'func1')
            out1 = input1 + fb
            annotate.output_tensors({'out1': out1}, 'func1', export_with="torch")
            
            # func2: middle of the chain
            in2 = annotate.input_tensors({'in2': out1}, 'func2')
            out2 = in2 * 2.0
            annotate.output_tensors({'out2': out2}, 'func2', export_with="torch")
            
            # func3: end of chain, output feeds back to func1
            in3 = annotate.input_tensors({'in3': out2}, 'func3')
            out3 = in3 - 1.0
            annotate.output_tensors({'out3': out3}, 'func3', export_with="torch")
            
            feedback_tensor = out3
        
        annotate.stop()
        annotate.compile_graph(visualize=False)
        
        # 3 nodes, 1 external input, 0 external outputs (all internal/feedback)
        # 2 internal connections (func1->func2, func2->func3)
        # 1 feedback connection (func3->func1)
        self.verify_num_connections(
            annotate, nodes=3, inputs=1, outputs=0,
            internal_connections=2, feedback_connections=1)
        self.verify_all_models_exist('func1', 'func2', 'func3')

    def test_annotate_traced_tensors_with_complex_nested_io(self):
        '''Single node with complex nested dict/list IO run multiple times'''
        
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        for i in range(10):
            # Complex nested input structure
            tensor_list, single_tensor = annotate.input_tensors({
                'tensor_list': [
                    torch.tensor([1.0, 2.0, 3.0]),
                    torch.tensor([4.0, 5.0, 6.0])
                ],
                'single_tensor': torch.tensor([7.0, 8.0, 9.0])
            }, 'complex_node')
            
            # Process the nested inputs
            list_sum = tensor_list[0] + tensor_list[1]
            combined = list_sum + single_tensor
            
            # Complex nested output structure
            annotate.output_tensors({
                    'sum': list_sum,
                    'combined': combined}, 'complex_node', export_with="torch")
        
        annotate.stop()
        annotate.compile_graph(visualize=False)
        
        # 1 node, 3 flattened inputs, 2 flattened outputs
        self.verify_num_connections(
            annotate, nodes=1, inputs=3, outputs=2, internal_connections=0)
        self.verify_all_models_exist('complex_node')

    def test_annotate_traced_tensors_diamond_with_feedback(self):
        '''Diamond pattern: func1 splits into func2a and func2b, 
        which merge at func3. func3 feeds back to func1.
        
              -> func2a ->
        func1              func3 -> (feedback to func1)
              -> func2b ->
        '''
        
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        feedback_tensor = torch.tensor([0.0, 0.0, 0.0])
        
        for i in range(10):
            # func1: source node with feedback
            in1, fb = annotate.input_tensors({
                'input': torch.tensor([1.0, 2.0, 3.0]),
                'feedback': feedback_tensor
            }, 'func1')
            out1 = in1 + fb
            annotate.output_tensors({'out': out1}, 'func1', export_with="torch")
            
            # func2a: first parallel branch
            in2a = annotate.input_tensors({'input2a': out1}, 'func2a')
            out2a = in2a * 2.0
            annotate.output_tensors({'out': out2a}, 'func2a', export_with="torch")
            
            # func2b: second parallel branch
            in2b = annotate.input_tensors({'input2b': out1}, 'func2b')
            out2b = in2b * 3.0
            annotate.output_tensors({'out': out2b}, 'func2b', export_with="torch")
            
            # func3: merge node, output feeds back
            in3a, in3b = annotate.input_tensors({
                'in_a': out2a,
                'in_b': out2b
            }, 'func3')
            out3 = in3a + in3b
            annotate.output_tensors({'out': out3}, 'func3', export_with="torch")
            
            feedback_tensor = out3
        
        annotate.stop()
        annotate.compile_graph(visualize=True)

        # 4 nodes, 1 external input
        # 3 internal connections (func1->func2a, func1->func2b, func2a->func3, func2b->func3)
        # Actually func2a->func3 and func2b->func3 are 2 separate connections
        # 1 feedback connection (func3->func1)
        self.verify_num_connections(
            annotate, nodes=4, inputs=1, outputs=0,
            internal_connections=4, feedback_connections=1)
        self.verify_all_models_exist('func1', 'func2a', 'func2b', 'func3')

    def test_annotate_traced_tensors_two_independent_feedback_loops(self):
        '''Two independent feedback loops running in parallel.
        Loop A: funcA1 -> funcA2 -> (feedback to funcA1)
        Loop B: funcB1 -> funcB2 -> (feedback to funcB1)
        '''
        
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        feedback_a = torch.tensor([0.0, 0.0, 0.0])
        feedback_b = torch.tensor([0.0, 0.0, 0.0])
        
        for i in range(10):
            # Loop A
            inA1, fbA = annotate.input_tensors({
                'input': torch.tensor([1.0, 2.0, 3.0]),
                'feedback': feedback_a
            }, 'funcA1')
            outA1 = inA1 + fbA
            annotate.output_tensors({'out': outA1}, 'funcA1', export_with="torch")
            
            inA2 = annotate.input_tensors({'inputA2': outA1}, 'funcA2')
            outA2 = inA2 * 2.0
            annotate.output_tensors({'out': outA2}, 'funcA2', export_with="torch")
            feedback_a = outA2
            
            # Loop B (independent)
            inB1, fbB = annotate.input_tensors({
                'input': torch.tensor([4.0, 5.0, 6.0]),
                'feedback': feedback_b
            }, 'funcB1')
            outB1 = inB1 - fbB
            annotate.output_tensors({'out': outB1}, 'funcB1', export_with="torch")
            
            inB2 = annotate.input_tensors({'inputB2': outB1}, 'funcB2')
            outB2 = inB2 / 2.0
            annotate.output_tensors({'out': outB2}, 'funcB2', export_with="torch")
            feedback_b = outB2
        
        annotate.stop()
        annotate.compile_graph(visualize=False)
        
        # 4 nodes, 2 external inputs (one per loop)
        # 2 internal connections (funcA1->funcA2, funcB1->funcB2)
        # 2 feedback connections (funcA2->funcA1, funcB2->funcB1)
        self.verify_num_connections(
            annotate, nodes=4, inputs=2, outputs=0,
            internal_connections=2, feedback_connections=2)
        self.verify_all_models_exist('funcA1', 'funcA2', 'funcB1', 'funcB2')

    def test_annotate_traced_tensors_nested_dict_multiple_runs(self):
        '''Test nested dict input structure with multiple iterations.
        This tests that validation correctly handles nested structures on reentry.'''
        
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        for i in range(10):
            # Nested dict input - {'group': {'a': tensor, 'b': tensor}}
            inputs = annotate.input_tensors({
                'group': {
                    'x': torch.tensor([1.0, 2.0, 3.0]),
                    'y': torch.tensor([4.0, 5.0, 6.0])
                }
            }, 'nested_node')
            
            # Access nested structure
            result = inputs['x'] + inputs['y']
            
            annotate.output_tensors({'result': result}, 'nested_node', export_with="torch")
        
        annotate.stop()
        annotate.compile_graph(visualize=False)
        
        # 1 node, 2 flattened inputs (group_x, group_y), 1 output
        self.verify_num_connections(
            annotate, nodes=1, inputs=2, outputs=1, internal_connections=0)
        self.verify_all_models_exist('nested_node')

    def test_annotate_traced_tensors_nested_list_multiple_runs(self):
        '''Test nested list input structure with multiple iterations.'''
        
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        for i in range(10):
            # List input that gets flattened
            tensor_list = annotate.input_tensors({
                'tensors': [
                    torch.tensor([1.0, 2.0, 3.0]),
                    torch.tensor([4.0, 5.0, 6.0]),
                    torch.tensor([7.0, 8.0, 9.0])
                ]
            }, 'list_node')
            
            # Access list elements
            result = tensor_list[0] + tensor_list[1] + tensor_list[2]
            
            annotate.output_tensors({'result': result}, 'list_node', export_with="torch")
        
        annotate.stop()
        annotate.compile_graph(visualize=False)
        
        # 1 node, 3 flattened inputs, 1 output
        self.verify_num_connections(
            annotate, nodes=1, inputs=3, outputs=1, internal_connections=0)
        self.verify_all_models_exist('list_node')

    def test_annotate_traced_tensors_mixed_nested_structure_multiple_runs(self):
        '''Test mixed nested structure (dict containing list) with multiple iterations.'''
        
        annotate.start(name=self.TEST_GRAPH_NAME)
        
        for i in range(10):
            # Mixed nested: dict with both list and scalar tensor
            inputs = annotate.input_tensors({
                'batch': [
                    torch.tensor([1.0, 2.0]),
                    torch.tensor([3.0, 4.0])
                ],
                'scale': torch.tensor([2.0])
            }, 'mixed_node')
            
            # Access mixed structure - inputs should be (list, tensor) tuple
            batch_list, scale = inputs
            result = (batch_list[0] + batch_list[1]) * scale
            
            annotate.output_tensors({'result': result}, 'mixed_node', export_with="torch")
        
        annotate.stop()
        annotate.compile_graph(visualize=False)
        
        # 1 node, 3 flattened inputs (batch_0, batch_1, scale), 1 output
        self.verify_num_connections(
            annotate, nodes=1, inputs=3, outputs=1, internal_connections=0)
        self.verify_all_models_exist('mixed_node')


class TestAnnotateMixed(LEAPPFunctionalTestBase):
    def test_mixing_annotated_tensors_and_method_nodes(self):
        """Test: traced_tensors → method"""
        def run_function(tensor):
            return tensor + 1.0
        
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA - 2.0
        
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        annotate.start(name=self.TEST_GRAPH_NAME)
        input_tensor = annotate.input_tensors({'input_tensor': input_tensor}, 'run_function')
        output_tensor = run_function(input_tensor)
        annotate.output_tensors({'output_tensor': output_tensor}, 'run_function', export_with="torch")
        outputA = funcA(output_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=2, inputs=1, outputs=1, internal_connections=1)
        self.verify_all_models_exist('run_function', 'funcA')

    def test_method_then_traced_tensors(self):
        """Test: method → traced_tensors"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA * 2.0
        
        def run_function(tensor):
            return tensor + 10.0
        
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        annotate.start(name=self.TEST_GRAPH_NAME)
        output_funcA = funcA(input_tensor)
        # Now use traced tensors for the next step
        traced_input = annotate.input_tensors({'traced_input': output_funcA}, 'run_function')
        output_tensor = run_function(traced_input)
        annotate.output_tensors({'output_tensor': output_tensor}, 'run_function', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=2, inputs=1, outputs=1, internal_connections=1)
        self.verify_all_models_exist('funcA', 'run_function')

    def test_traced_tensors_sandwich_method(self):
        """Test: traced_tensors → method → traced_tensors"""
        def preprocess(tensor):
            return tensor * 2.0
        
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA + 5.0
        def postprocess(tensor):
            return tensor - 1.0
        
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        annotate.start(name=self.TEST_GRAPH_NAME)
        # First traced tensor node
        traced_input = annotate.input_tensors({'input': input_tensor}, 'preprocess')
        preprocessed = preprocess(traced_input)
        annotate.output_tensors({'preprocessed': preprocessed}, 'preprocess', export_with="torch")
        # Method in the middle
        method_output = funcA(preprocessed)
        # Second traced tensor node
        traced_method_output = annotate.input_tensors({'method_out': method_output}, 'postprocess')
        postprocessed = postprocess(traced_method_output)
        annotate.output_tensors({'postprocessed': postprocessed}, 'postprocess', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=3, inputs=1, outputs=1, internal_connections=2)
        self.verify_all_models_exist('preprocess', 'funcA', 'postprocess')

    def test_block_then_traced_tensors(self):
        """Test: block → traced_tensors"""
        def run_function(tensor):
            return tensor + 100.0
        
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        annotate.start(name=self.TEST_GRAPH_NAME)
        # Block context first
        with annotate.block('block_node', inputs=['input_tensor'], outputs=['block_output'], export_with="torch"):
            block_output = input_tensor * 3.0
        # Then traced tensors
        traced_input = annotate.input_tensors({'traced_input': block_output}, 'run_function')
        output_tensor = run_function(traced_input)
        annotate.output_tensors({'output_tensor': output_tensor}, 'run_function', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=2, inputs=1, outputs=1, internal_connections=1)
        self.verify_all_models_exist('block_node', 'run_function')

    def test_traced_tensors_then_block(self):
        """Test: traced_tensors → block"""
        def run_function(tensor):
            return tensor * 2.0
        
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        annotate.start(name=self.TEST_GRAPH_NAME)
        # Traced tensors first
        traced_input = annotate.input_tensors({'input': input_tensor}, 'run_function')
        output_tensor = run_function(traced_input)
        annotate.output_tensors({'output': output_tensor}, 'run_function', export_with="torch")
        # Then block context
        with annotate.block('block_node', inputs=['output_tensor'], outputs=['block_output'], export_with="torch"):
            block_output = output_tensor + 50.0
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=2, inputs=1, outputs=1, internal_connections=1)
        self.verify_all_models_exist('run_function', 'block_node')

    def test_method_then_block_then_traced_tensors(self):
        """Test: method → block → traced_tensors"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA * 2.0
        
        def run_function(tensor):
            return tensor - 5.0
        
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        annotate.start(name=self.TEST_GRAPH_NAME)
        # Method first
        method_output = funcA(input_tensor)
        # Then block
        with annotate.block('block_node', inputs=['method_output'], outputs=['block_output'], export_with="torch"):
            block_output = method_output + 10.0
        # Then traced tensors
        traced_input = annotate.input_tensors({'traced_input': block_output}, 'run_function')
        output_tensor = run_function(traced_input)
        annotate.output_tensors({'output_tensor': output_tensor}, 'run_function', export_with="torch")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=3, inputs=1, outputs=1, internal_connections=2)
        self.verify_all_models_exist('funcA', 'block_node', 'run_function')

    def test_two_parallel_traced_tensors_merge_to_method(self):
        """Test: two parallel traced_tensor nodes feeding into one method"""
        def process_a(tensor):
            return tensor * 2.0
        
        def process_b(tensor):
            return tensor * 3.0
        
        @annotate.method(export_with="torch")
        def combine(inputA: torch.Tensor, inputB: torch.Tensor):
            return inputA + inputB
        
        input_a = torch.tensor([1.0, 2.0, 3.0])
        input_b = torch.tensor([4.0, 5.0, 6.0])
        annotate.start(name=self.TEST_GRAPH_NAME)
        # First traced tensor path
        traced_a = annotate.input_tensors({'input_a': input_a}, 'process_a')
        # First traced tensor path
        traced_b = annotate.input_tensors({'input_b': input_b}, 'process_b')

        output_a = process_a(traced_a)
        output_b = process_b(traced_b)

        annotate.output_tensors({'output_a': output_a}, 'process_a', export_with="torch")
        annotate.output_tensors({'output_b': output_b}, 'process_b', export_with="torch")
        # Combine with method
        combined = combine(output_a, output_b)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=3, inputs=2, outputs=1, internal_connections=2)
        self.verify_all_models_exist('process_a', 'process_b', 'combine')


if __name__ == '__main__':
    unittest.main(verbosity=2)
