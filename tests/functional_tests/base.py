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
import unittest
import os
import shutil
import torch
from safetensors.torch import load_file
from leapp.export_manager import ExportManager
from leapp.leapp import _MANAGER as annotate # non-protected access to annotate singleton
import yaml


class LEAPPFunctionalTestBase(unittest.TestCase):
    def setUp(self):
        self.TEST_GRAPH_NAME = "test_graph"

    def tearDown(self):
        if os.path.exists(self.TEST_GRAPH_NAME):
            shutil.rmtree(self.TEST_GRAPH_NAME)
        if ExportManager.is_interpret_graph_enabled():
            ExportManager.set_interpret_graph(False)
        annotate.reset_nodes()
        annotate.set_dry_run_and_non_traced(False, [])
    
    def verify_all_models_exist(self, *model_names):
        for model_name in model_names:
            model_exists = False
            model_exists |= os.path.exists(os.path.join(self.TEST_GRAPH_NAME, f"{model_name}.pt"))
            model_exists |= os.path.exists(os.path.join(self.TEST_GRAPH_NAME, f"{model_name}.onnx"))
            self.assertTrue(model_exists,
                            f"Model {model_name} does not exist")

    def verify_num_connections(self, leapp_annotation, nodes=None, inputs=None, outputs=None,
                               internal_connections=None, feedback_connections=None):
        if nodes is not None:
            self.assertEqual(nodes, len(leapp_annotation.detected_nodes),
                             "Number of nodes do not match")
        if inputs is not None:
            total_inputs = sum(
                [len(graph_inputs) for graph_inputs in leapp_annotation.detected_pipeline['inputs'].values()])
            self.assertEqual(inputs, total_inputs,
                             "Number of inputs do not match")
        if outputs is not None:
            total_outputs = sum(
                [len(graph_outputs) for graph_outputs in leapp_annotation.detected_pipeline['outputs'].values()])
            self.assertEqual(outputs, total_outputs,
                             "Number of outputs do not match")
        if internal_connections is not None:
            num_connections = 0
            for _, targets in leapp_annotation.detected_pipeline['data_flow'].items():
                num_connections += len(targets)
            self.assertEqual(internal_connections, num_connections, "Number of internal connections do not match")
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
            
            # Check if the output is a single tuple - if so, unwrap it
            if len(outputs) == 1:
                output_type = outputs[0].type()
                # Check if it's a tuple type by looking at the string representation
                type_str = str(output_type)
                if type_str.startswith('Tuple[') or type_str.startswith('('):
                    # It's a tuple - unwrap it by getting the element types
                    if hasattr(output_type, 'elements'):
                        # This is a TupleType, get the individual elements
                        tuple_elements = output_type.elements()
                        for i, elem_type in enumerate(tuple_elements):
                            output_info = {
                                'index': i,
                                'debug_name': f'tuple_element_{i}',
                                'type': str(elem_type),
                            }
                            
                            # Try to get shape information if it's a tensor
                            if hasattr(elem_type, 'sizes'):
                                try:
                                    output_info['shape'] = list(elem_type.sizes())
                                except Exception:
                                    output_info['shape'] = None
                            else:
                                output_info['shape'] = None
                            
                            result['outputs'].append(output_info)
                    else:
                        # Fallback: treat as single output
                        output_info = {
                            'index': 0,
                            'debug_name': outputs[0].debugName(),
                            'type': type_str,
                            'shape': None
                        }
                        result['outputs'].append(output_info)
                else:
                    # Not a tuple, process normally
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
            else:
                # Multiple outputs, process normally
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
        from leapp.leapp_graph.datatypes.traced_data import TracedData
        if isinstance(source_data, TracedData):
            source_data = source_data.data
        if isinstance(target_data, TracedData):
            target_data = target_data.data

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

    def _flatten_to_tensors(self, data):
        """
        Recursively flatten nested structures (lists, dicts) into a flat list of tensors.
        Uses the same traversal order as describe_io_helper.
        """
        tensors = []
        
        def _flatten(item):
            if isinstance(item, torch.Tensor):
                tensors.append(item)
            elif isinstance(item, dict):
                for value in item.values():
                    _flatten(value)
            elif isinstance(item, (list, tuple)):
                for elem in item:
                    _flatten(elem)
        
        _flatten(data)
        return tensors

    def _assert_artifacts(self, save_dir, graph_name):
        yaml_path = os.path.join(save_dir, f"{graph_name}.yaml")
        model_path = os.path.join(save_dir, "identity.pt")
        self.assertTrue(os.path.exists(yaml_path),
                        f"YAML not found at {yaml_path}")
        self.assertTrue(os.path.exists(model_path),
                        f"Model not found at {model_path}")
        # The exported model_path entry in YAML must be relative to the YAML dir.
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        identity_params = config["models"]["identity"]["parameters"]
        self.assertEqual(identity_params["model_path"], "identity.pt",
                         f"model_path should be relative to YAML directory, "
                         f"got {identity_params['model_path']!r}")

    def verify_feedback_initial_values(self, expected_values, graph_name=None):
        """Verify that the feedback initial values safetensors file exists and contains expected values.

        Args:
            expected_values: Dict mapping key names (e.g. "policy/counter") to expected tensors.
                Values can be:
                - torch.Tensor: exact value match
                - tuple: expected shape only (e.g. (1, 3, 4)) — useful when values are random
            graph_name: Optional graph name. Defaults to TEST_GRAPH_NAME.
        """
        if graph_name is None:
            graph_name = self.TEST_GRAPH_NAME
        safetensors_path = os.path.join(graph_name, f"{graph_name}_initial_values.safetensors")
        self.assertTrue(os.path.exists(safetensors_path),
                        f"Feedback initial values safetensors file not found at {safetensors_path}")
        loaded = load_file(safetensors_path)

        for key, expected in expected_values.items():
            self.assertIn(key, loaded, f"Key '{key}' not found in feedback initial values")
            if isinstance(expected, tuple):
                # Shape-only check
                self.assertEqual(loaded[key].shape, torch.Size(expected),
                                 f"Shape mismatch for '{key}': got {loaded[key].shape}, expected {expected}")
            else:
                # Exact value check
                self.assertTrue(torch.equal(loaded[key], expected),
                                f"Value mismatch for '{key}': got {loaded[key]}, expected {expected}")

    def verify_safetensors_matches_feedback(self, leapp_annotation, graph_name=None):
        """Verify safetensors keys exactly match the feedback targets in the pipeline.

        When feedback_flow is non-empty, a safetensors file must exist and its
        keys must correspond 1-to-1 with the feedback target inputs. When
        feedback_flow is empty, the safetensors file must NOT exist.
        """
        if graph_name is None:
            graph_name = self.TEST_GRAPH_NAME
        safetensors_path = os.path.join(
            graph_name, f"{graph_name}_initial_values.safetensors")

        feedback_flow = leapp_annotation.detected_pipeline.get(
            'feedback_flow', {})

        # Collect all feedback target keys (node_name/input_name)
        expected_keys = set()
        for targets in feedback_flow.values():
            for target in targets:
                expected_keys.add(target)

        if not expected_keys:
            self.assertFalse(
                os.path.exists(safetensors_path),
                "No feedback connections exist, but safetensors file was created")
            return

        self.assertTrue(
            os.path.exists(safetensors_path),
            f"Feedback connections exist but safetensors file not found at {safetensors_path}")

        loaded = load_file(safetensors_path)
        actual_keys = set(loaded.keys())

        self.assertEqual(
            expected_keys, actual_keys,
            f"Safetensors keys do not match feedback targets.\n"
            f"  Expected: {sorted(expected_keys)}\n"
            f"  Actual:   {sorted(actual_keys)}")

    def verify_inference_manager(self, source_inputs, source_outputs,
                                 graph_name=None, rtol=1e-3, atol=1e-5):
        """Load the exported graph via InferenceManager, run it, and compare outputs.

        Args:
            source_inputs: Dict mapping 'node/input' to tensors used during tracing.
            source_outputs: Dict mapping 'node/output' to expected output tensors.
            graph_name: Override for TEST_GRAPH_NAME.
            rtol: Relative tolerance for allclose.
            atol: Absolute tolerance for allclose.
        """
        from leapp import InferenceManager

        if graph_name is None:
            graph_name = self.TEST_GRAPH_NAME
        yaml_path = os.path.join(graph_name, f"{graph_name}.yaml")
        self.assertTrue(os.path.exists(yaml_path),
                        f"YAML not found at {yaml_path}")

        manager = InferenceManager(yaml_path)

        # Verify expected input/output keys exist
        for key in source_inputs:
            self.assertIn(key, manager.inputs,
                          f"Input key '{key}' not found in InferenceManager inputs: {manager.inputs}")
        for key in source_outputs:
            self.assertIn(key, manager.outputs,
                          f"Output key '{key}' not found in InferenceManager outputs: {manager.outputs}")

        # Move inputs to the device the InferenceManager expects
        device_inputs = {}
        for key, tensor in source_inputs.items():
            node_name = key.split('/')[0]
            target_device = manager.nodes[node_name].device
            device_inputs[key] = tensor.to(target_device)

        # Run inference
        exported_outputs = manager.run_policy(device_inputs)

        # Compare each expected output
        for key, expected in source_outputs.items():
            self.assertIn(key, exported_outputs,
                          f"Output key '{key}' missing from InferenceManager results")
            actual = exported_outputs[key]
            if expected.device != actual.device:
                actual = actual.to(expected.device)
            self.assertTrue(
                torch.allclose(expected, actual, rtol=rtol, atol=atol),
                f"Output mismatch for '{key}':\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual}\n"
                f"  Max diff: {(expected - actual).abs().max().item():.6e}")

    def verify_single_torchscript_model_expected_value(self, inputs, expected_outputs, model_name, model_path=None):
        if model_path is None:
            model_path = self.TEST_GRAPH_NAME
        model = torch.jit.load(os.path.join(model_path, f"{model_name}.pt"))
        
        # Flatten all inputs since models now expect flat tensor arguments
        flat_inputs = []
        for inp in inputs:
            flat_inputs.extend(self._flatten_to_tensors(inp))
        flat_expected_outputs = []
        for out in expected_outputs:
            flat_expected_outputs.extend(self._flatten_to_tensors(out))
        flat_expected_outputs = tuple(flat_expected_outputs)
        outputs = model(*flat_inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        for output, expected_output in zip(outputs, flat_expected_outputs):
            self.assertTrue(self.verify_data_exact_match(output, expected_output), "An output value does not match expected value: "
                            f"got {output} but expected {expected_output}")
