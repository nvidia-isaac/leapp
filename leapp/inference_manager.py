#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import json
import yaml

import torch

from safetensors.torch import load_file


from leapp.utils.tensor_description import map_to_torch_dtype, validate_connection_compatibility

from leapp.backends.torch_export_backend import TorchExportBackend
from leapp.backends.onnx_export_backend import ONNXExportBackend


class NodeManager:
    def __init__(self, name, model_path, inputs, outputs, parameters):
        self.name = name
        self.model_path = model_path
        self.input_descriptions = inputs
        self.output_descriptions = outputs


        backend = self._create_backend(parameters['backend'])
        backend.load(model_path, parameters['sha256sum'])
        self.model = backend.compiled_model

        self.device = backend.runtime_device

    @property
    def input_names(self):
        return [input['name'] for input in self.input_descriptions]

    @property
    def input_shapes(self):
        return [input['shape'] for input in self.input_descriptions]

    @property
    def input_dtypes(self):
        return [input['dtype'] for input in self.input_descriptions]

    @property
    def output_names(self):
        return [output['name'] for output in self.output_descriptions]

    @property
    def output_shapes(self):
        return [output['shape'] for output in self.output_descriptions]

    @property
    def output_dtypes(self):
        return [output['dtype'] for output in self.output_descriptions]

    @property
    def mock_input(self):
        input_values_to_populate = {}
        for input_val in self.input_descriptions:
            input_name = input_val['name']
            input_shape = json.loads(input_val['shape']) if isinstance(
                input_val['shape'], str) else input_val['shape']
            input_dtype = input_val['dtype']
            input_value = torch.zeros(
                tuple(input_shape), dtype=map_to_torch_dtype(input_dtype))
            input_values_to_populate[input_name] = input_value.to(self.device)

        return input_values_to_populate

    @property
    def mock_output(self):
        output_values_to_populate = {}
        for output_val in self.output_descriptions:
            output_name = output_val['name']
            output_shape = json.loads(output_val['shape']) if isinstance(
                output_val['shape'], str) else output_val['shape']
            output_dtype = output_val['dtype']
            output_value = torch.zeros(
                tuple(output_shape), dtype=map_to_torch_dtype(output_dtype))
            output_values_to_populate[output_name] = output_value.to(
                self.device)

        return output_values_to_populate

    def _create_backend(self, backend):
        """utilizes the backends just to load the model. the backend is not used for compilation or saving."""
        if backend == "jit":
            return TorchExportBackend(None)
        elif backend == "onnx":
            return ONNXExportBackend(None)
        elif backend == "warp":
            # Lazy import so warp stays an optional dependency.
            from leapp.backends.warp_export_backend import WarpExportBackend
            return WarpExportBackend(None)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def __call__(self, *inputs):
        if len(inputs) != len(self.input_descriptions):
            raise ValueError(
                f"Expected {len(self.input_descriptions)} inputs, got {len(inputs)}")

        outputs = self.model(*inputs)

        # Handle single output case - unwrap from tuple if needed
        if len(self.output_descriptions) == 1:
            if isinstance(outputs, tuple):
                return outputs[0]
            return outputs
        # Handle multiple outputs
        elif len(outputs) == len(self.output_descriptions):
            return outputs
        else:
            raise ValueError(
                f"Expected {len(self.output_descriptions)} outputs, got {len(outputs)}")


class InferenceManager:
    def __init__(self, model_path):
        # data reading variables
        self.models = None
        self.pipeline = None
        self.system_info = None

        # runtime variables

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Leapp description file not found at {model_path}")

        if not model_path.endswith(".yaml"):
            raise ValueError(
                f"Leapp description file must end with .yaml, got {model_path}")

        self.model_path = model_path

        self._load_description()

        self.nodes = self._create_nodes()

        self._validate_and_create_inference_graph()

        self._prepopulate_feedback_inputs()

    def _load_description(self):
        with open(self.model_path, "r") as f:
            self.leapp_description = yaml.safe_load(f)

        if any(key not in self.leapp_description for key in ["models", "pipeline", "system information"]):
            raise ValueError(
                f"Leapp description file must contain models, pipeline, and system_info, got {self.leapp_description.keys()}")

        self.models = self.leapp_description["models"]
        self.pipeline = self.leapp_description["pipeline"]
        self.system_info = self.leapp_description["system information"]

    def _create_nodes(self):
        nodes = {}
        for name, description in self.models.items():
            if any(key not in description for key in ["inputs", "outputs", "parameters"]):
                raise ValueError(
                    f"Model description must contain inputs, outputs, and parameters, got {description.keys()}")
            # load the model
            parameters = description['parameters']
            if any(key not in parameters for key in ["model_path", "md5sum", "sha256sum", "backend"]):
                raise ValueError(
                    f"Model description must contain model_path, md5sum, sha256sum, and backend, got {parameters.keys()}")

            model_path = parameters['model_path']
            base_path = os.path.dirname(self.model_path)

            relative_model_path = os.path.join(base_path, model_path)
            if os.path.exists(relative_model_path):
                model_path = relative_model_path

            node = NodeManager(
                name, model_path, description['inputs'], description['outputs'], description['parameters'])
            nodes[name] = node

        return nodes

    def _validate_and_create_inference_graph(self):
        self.value_dict = {}

        # inputs - preallocate tensors for each node's inputs
        for node_name, node in self.nodes.items():
            self.value_dict[node_name] = node.mock_input

        self.organized_pipeline_connections = {}
        # Merge data_flow and feedback_flow, combining target lists for shared keys
        all_flows = {}
        for flow_dict in [self.pipeline.get('data_flow', {}), self.pipeline.get('feedback_flow', {})]:
            for source, targets in flow_dict.items():
                if source in all_flows:
                    all_flows[source] = all_flows[source] + targets
                else:
                    all_flows[source] = targets
        # organize the pipeline connections
        for source, targets in all_flows.items():
            source_node_name, source_output_name = source.split('/')
            if source_node_name not in self.organized_pipeline_connections:
                self.organized_pipeline_connections[source_node_name] = {}
            self.organized_pipeline_connections[source_node_name][source_output_name] = [
            ]
            for target in targets:
                target_node_name, target_input_name = target.split('/')
                self.organized_pipeline_connections[source_node_name][source_output_name].append(
                    (target_node_name, target_input_name))

        # Create '==out==' slot and route final outputs to it
        output_cache = {}
        for node_name, output_names in self.pipeline['outputs'].items():
            node = self.nodes[node_name]
            # Get output descriptions for this node
            output_descs = {desc['name']
                : desc for desc in node.output_descriptions}

            for output_name in output_names:
                desc = output_descs[output_name]
                output_shape = json.loads(desc['shape']) if isinstance(
                    desc['shape'], str) else desc['shape']
                output_dtype = map_to_torch_dtype(desc['dtype'])
                # Create cache tensor with unique key: node_name/output_name
                cache_key = f"{node_name}/{output_name}"
                output_cache[cache_key] = torch.zeros(
                    tuple(output_shape), dtype=output_dtype, device=node.device)

                # Add connection from this output to ==out==
                if node_name not in self.organized_pipeline_connections:
                    self.organized_pipeline_connections[node_name] = {}
                if output_name not in self.organized_pipeline_connections[node_name]:
                    self.organized_pipeline_connections[node_name][output_name] = [
                    ]
                self.organized_pipeline_connections[node_name][output_name].append(
                    ('==out==', cache_key))

        self.value_dict['==out=='] = output_cache

        self._validate_output_routing()

        # Validate shape and dtype compatibility for all pipeline connections.
        self._validate_connection_compatibility()

    def _validate_output_routing(self):
        """Ensure every model output is routed somewhere before inference starts."""
        for node_name, node in self.nodes.items():
            pipeline_map = self.organized_pipeline_connections.get(node_name, {})
            missing_outputs = [
                output_name for output_name in node.output_names
                if output_name not in pipeline_map
            ]
            if missing_outputs:
                raise ValueError(
                    f"Node '{node_name}' has unroutable outputs: {missing_outputs}. "
                    "Every model output must appear in pipeline data_flow, "
                    "feedback_flow, or pipeline outputs. This may indicate the "
                    "YAML was edited manually and is inconsistent with the exported model."
                )

    def _validate_connection_compatibility(self):
        all_flows = {}
        for flow_key in ('data_flow', 'feedback_flow'):
            for source, targets in self.pipeline.get(flow_key, {}).items():
                if source in all_flows:
                    all_flows[source] = all_flows[source] + targets
                else:
                    all_flows[source] = list(targets)

        for source, targets in all_flows.items():
            source_node_name, source_output_name = source.split('/')

            # Validate source node exists
            if source_node_name not in self.nodes:
                raise ValueError(
                    f"Source node '{source_node_name}' not found in models")

            source_node = self.nodes[source_node_name]

            # Validate source output exists
            source_output_names = [desc['name']
                                   for desc in source_node.output_descriptions]
            if source_output_name not in source_output_names:
                raise ValueError(
                    f"Source output '{source_output_name}' not found in node '{source_node_name}'. "
                    f"Available outputs: {source_output_names}"
                )

            # Get source output description
            source_desc = next(
                desc for desc in source_node.output_descriptions
                if desc['name'] == source_output_name
            )
            source_shape = json.loads(source_desc['shape']) if isinstance(
                source_desc['shape'], str) else source_desc['shape']
            source_dtype = map_to_torch_dtype(source_desc['dtype'])

            for target in targets:
                target_node_name, target_input_name = target.split('/')

                # Validate target node exists
                if target_node_name not in self.nodes:
                    raise ValueError(
                        f"Target node '{target_node_name}' not found in models")

                # Validate target input exists
                if target_input_name not in self.value_dict[target_node_name]:
                    available_inputs = list(
                        self.value_dict[target_node_name].keys())
                    raise ValueError(
                        f"Target input '{target_input_name}' not found in node '{target_node_name}'. "
                        f"Available inputs: {available_inputs}"
                    )

                target_tensor = self.value_dict[target_node_name][target_input_name]

                validate_connection_compatibility(
                    source_name=source,
                    source_shape=source_shape,
                    source_dtype=source_dtype,
                    target_name=target,
                    target_shape=target_tensor.shape,
                    target_dtype=target_tensor.dtype,
                )

    def _prepopulate_feedback_inputs(self):
        """Load feedback initial values from safetensors and populate input buffers.
        
        If the pipeline has an 'initial_values' field pointing to a safetensors file,
        load it and overwrite the corresponding feedback input buffers so deployers
        don't need to know what to initialize them as.
        """
        initial_values_file = self.pipeline.get('initial_values')
        if not initial_values_file:
            return

        base_path = os.path.dirname(self.model_path)
        safetensors_path = os.path.join(base_path, initial_values_file)

        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(
                f"Feedback initial values file not found at {safetensors_path}")

        initial_values = load_file(safetensors_path)

        for key, tensor in initial_values.items():
            node_name, input_name = key.split('/')
            if node_name in self.value_dict and input_name in self.value_dict[node_name]:
                device = self.value_dict[node_name][input_name].device
                self.value_dict[node_name][input_name] = tensor.to(device)
            else:
                raise ValueError(
                    f"Feedback initial value key '{key}' does not match any node input. "
                    f"Available nodes: {list(self.value_dict.keys())}")

    def run_policy(self, inputs: dict[str, torch.Tensor]):
        # Update input tensors with provided values (keys are "node_name/input_name")
        for key, input_value in inputs.items():
            try:
                node_name, input_name = key.split('/')
            except ValueError:
                raise ValueError(
                    f"Invalid input key: {key}\n Expected format: node_name/input_name")
            self.value_dict[node_name][input_name] = input_value

        for node_name, node in self.nodes.items():
            # Run inference via node's __call__
            # Use node.input_names to ensure inputs are passed in the correct order
            node_inputs = [self.value_dict[node_name][name]
                      for name in node.input_names]
            
            outputs = node(*node_inputs)
            output_order = node.output_names
            # Ensure outputs is always a tuple/list for consistent iteration
            if len(output_order) == 1:
                outputs = (outputs,)

            pipeline_map = self.organized_pipeline_connections[node_name]
            for output_name, output_value in zip(output_order, outputs):
                targets = pipeline_map[output_name]
                for i, (target_node_name, target_input_name) in enumerate(targets):
                    if i == 0:
                        # Zero-copy: direct assignment for first consumer
                        self.value_dict[target_node_name][target_input_name] = output_value
                    else:
                        # Clone for 2nd+ consumers to prevent data corruption
                        self.value_dict[target_node_name][target_input_name] = output_value.clone(
                        )

        return dict(self.value_dict['==out=='])

    @property
    def inputs(self) -> list:
        """Returns list of expected input keys in 'node_name/input_name' format."""
        input_keys = []
        for node_name, input_names in self.pipeline['inputs'].items():
            for input_name in input_names:
                input_keys.append(f"{node_name}/{input_name}")
        return input_keys

    @property
    def outputs(self) -> list:
        """Returns list of output keys in 'node_name/output_name' format."""
        output_keys = []
        for node_name, output_names in self.pipeline['outputs'].items():
            for output_name in output_names:
                output_keys.append(f"{node_name}/{output_name}")
        return output_keys

    @property
    def feedback_inputs(self) -> list:
        keys = []
        for targets in self.pipeline.get('feedback_flow', {}).values():
            keys.extend(targets)
        return keys

    def set_input_value(self, node_name: str, input_name: str, value: torch.Tensor):
        self.value_dict[node_name][input_name].copy_(value)

    def get_mock_input(self):
        """Generate random tensors for all external pipeline inputs.

        Returns:
            dict: Mapping of 'node_name/input_name' to random tensors with correct dtype, device, and shape.
        """
        mock_inputs = {}

        for node_name, input_names in self.pipeline['inputs'].items():
            node = self.nodes[node_name]
            # Create a lookup for input descriptions by name
            input_descs = {desc['name']
                : desc for desc in node.input_descriptions}

            for input_name in input_names:
                desc = input_descs[input_name]
                shape = json.loads(desc['shape']) if isinstance(
                    desc['shape'], str) else desc['shape']
                dtype = map_to_torch_dtype(desc['dtype'])
                device = node.device

                # Generate random tensor (randn for floats, randint for integers)
                if dtype.is_floating_point:
                    tensor = torch.randn(
                        tuple(shape), dtype=dtype, device=device)
                else:
                    tensor = torch.randint(0, 256, tuple(
                        shape), dtype=dtype, device=device)
                mock_inputs[f"{node_name}/{input_name}"] = tensor

        return mock_inputs
    
    def reset(self):
        """resets the inference manager to its initial state
        
        This method does not reset the internal state of the nodes, only the input and output buffers.
        It uses default initial values for feedback inputs.
        """
        with torch.no_grad():
            for buffer_group in self.value_dict.values():
                for tensor in buffer_group.values():
                    tensor.zero_()

        self._prepopulate_feedback_inputs()


    def __call__(self, inputs: dict[str, torch.Tensor]):
        return self.run_policy(inputs)
