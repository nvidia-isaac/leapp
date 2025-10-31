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
import os
import shutil
import torch


class LEAPPFunctionalTestBase(unittest.TestCase):
    def setUp(self):
        self.TEST_GRAPH_NAME = "test_graph"

    def tearDown(self):
        if os.path.exists(self.TEST_GRAPH_NAME):
            shutil.rmtree(self.TEST_GRAPH_NAME)

    def verify_num_connections(self, leapp_annotation, nodes=None, inputs=None, outputs=None,
                               internal_connections=None, feedback_connections=None):
        if nodes is not None:
            self.assertEqual(nodes, len(leapp_annotation.detected_nodes),
                             "Number of nodes do not match")
        if inputs is not None:
            total_inputs = sum(
                [len(graph_inputs) for graph_inputs in leapp_annotation.detected_pipeline['dangling_inputs'].values()])
            self.assertEqual(inputs, total_inputs,
                             "Number of inputs do not match")
        if outputs is not None:
            total_outputs = sum(
                [len(graph_outputs) for graph_outputs in leapp_annotation.detected_pipeline['dangling_outputs'].values()])
            self.assertEqual(outputs, total_outputs,
                             "Number of outputs do not match")
        if internal_connections is not None:
            self.assertEqual(internal_connections, len(
                leapp_annotation.detected_pipeline['data_flow']), "Number of internal connections do not match")
        if feedback_connections is not None:
            self.assertEqual(feedback_connections, len(
                leapp_annotation.detected_pipeline['feedback_flow']), "Number of feedback connections do not match")

    def inspect_torchscript_model(self, model_name, model_path=None):
        """
        Extract input and output information from a loaded TorchScript model.

        Args:
            model_name: Name of the model (without .pt extension)
            model_path: Optional path to the model directory. Defaults to TEST_GRAPH_NAME.

        Returns:
            dict: Dictionary containing:
                - 'inputs': List of dicts with input information (name, type, shape)
                - 'outputs': List of dicts with output information (name, type, shape)
                - 'graph': The computation graph object
                - 'code': String representation of the model's code
        """
        if model_path is None:
            model_path = self.TEST_GRAPH_NAME
        model = torch.jit.load(os.path.join(model_path, f"{model_name}.pt"))
        result = {
            'inputs': [],
            'outputs': [],
            'graph': None,
            'code': None
        }

        try:
            # Get the computation graph
            graph = model.graph
            result['graph'] = graph

            # Extract input information
            inputs = list(graph.inputs())
            for i, inp in enumerate(inputs):
                input_info = {
                    'index': i,
                    'debug_name': inp.debugName(),
                    'type': str(inp.type()),
                }

                # Try to get shape information if it's a tensor
                if hasattr(inp.type(), 'sizes'):
                    try:
                        input_info['shape'] = list(inp.type().sizes())
                    except Exception:
                        input_info['shape'] = None
                else:
                    input_info['shape'] = None

                result['inputs'].append(input_info)

            # Extract output information
            outputs = list(graph.outputs())
            for i, out in enumerate(outputs):
                output_info = {
                    'index': i,
                    'debug_name': out.debugName(),
                    'type': str(out.type()),
                }

                # Try to get shape information if it's a tensor
                if hasattr(out.type(), 'sizes'):
                    try:
                        output_info['shape'] = list(out.type().sizes())
                    except Exception:
                        output_info['shape'] = None
                else:
                    output_info['shape'] = None

                result['outputs'].append(output_info)

            # Get the code representation if available
            if hasattr(model, 'code'):
                result['code'] = model.code

        except Exception as e:
            print(f"Error inspecting TorchScript model: {e}")

        return result

    def print_torchscript_model_info(self, model_name, model_path=None, verbose=True):
        """
        Pretty print information about a TorchScript model.

        Args:
            model_name: Name of the model (without .pt extension)
            model_path: Optional path to the model directory. Defaults to TEST_GRAPH_NAME.
            verbose: If True, print the full graph representation
        """
        info = self.inspect_torchscript_model(model_name, model_path)

        print("\n" + "="*60)
        print("TorchScript Model Information")
        print("="*60)

        print("\n[Inputs]")
        if not info['inputs']:
            print("  No inputs found")
        else:
            for inp in info['inputs']:
                print(f"  Input {inp['index']}:")
                print(f"    Name: {inp['debug_name']}")
                print(f"    Type: {inp['type']}")
                if inp['shape']:
                    print(f"    Shape: {inp['shape']}")

        print("\n[Outputs]")
        if not info['outputs']:
            print("  No outputs found")
        else:
            for out in info['outputs']:
                print(f"  Output {out['index']}:")
                print(f"    Name: {out['debug_name']}")
                print(f"    Type: {out['type']}")
                if out['shape']:
                    print(f"    Shape: {out['shape']}")

        if verbose and info['graph']:
            print("\n[Computation Graph]")
            print(info['graph'])

        if info['code']:
            print("\n[Model Code]")
            print(info['code'])

        print("\n" + "="*60)

    def verify_data_exact_match(self, source_data, target_data):
        if type(source_data) is not type(target_data):
            return False

        if isinstance(source_data, torch.Tensor):
            if source_data.shape != target_data.shape:
                return False
            if source_data.dtype != target_data.dtype:
                return False
            if source_data.device != target_data.device:
                return False
            if not torch.equal(source_data, target_data):
                return False

        elif isinstance(source_data, list):
            if len(source_data) != len(target_data):
                return False
            for source_item, target_item in zip(source_data, target_data):
                if not self.verify_data_exact_match(source_item, target_item):
                    return False
        elif isinstance(source_data, dict):
            if source_data.keys() != target_data.keys():
                return False
            for key, source_item in source_data.items():
                if not self.verify_data_exact_match(source_item, target_data[key]):
                    return False
        else:
            if source_data != target_data:
                return False

        return True

    def verify_single_torchscript_model_expected_value(self, inputs, expected_outputs, model_name, model_path=None):
        if model_path is None:
            model_path = self.TEST_GRAPH_NAME
        model = torch.jit.load(os.path.join(model_path, f"{model_name}.pt"))
        outputs = model(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # if not isinstance(expected_outputs, tuple):
        #     expected_outputs = (expected_outputs,)
        for output, expected_output in zip(outputs, expected_outputs):
            self.assertTrue(self.verify_data_exact_match(output, expected_output), "An output value does not match expected value: "
                            f"got {output} but expected {expected_output}")
