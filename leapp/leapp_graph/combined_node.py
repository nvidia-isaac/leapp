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

from .leapp_node import LeappNode
from collections import OrderedDict
from typing import List, Dict, Tuple
import torch
from leapp._logging import _get_logger


class CombinedNodeModel(torch.nn.Module):
    """A combined model that chains multiple models together using pre-computed index routing.

    All routing is computed at __init__ time as integer indices, so forward() only 
    performs tensor operations and list indexing - no string lookups or Python functions.
    This makes it fully compatible with TorchScript tracing.
    """

    def __init__(self,
                 models: List[torch.nn.Module],
                 num_external_inputs: int,
                 model_input_indices: List[List[int]],
                 model_output_indices: List[List[int]],
                 final_output_indices: List[int],
                 num_slots: int):
        """
        Args:
            models: List of compiled models in execution order
            num_external_inputs: Number of external inputs to the combined model
            model_input_indices: For each model, list of slot indices to read inputs from
            model_output_indices: For each model, list of slot indices to write outputs to  
            final_output_indices: Slot indices for the final outputs
            num_slots: Total number of slots needed (external inputs + all intermediate values)
        """
        super().__init__()

        # Register models as submodules
        for idx, model in enumerate(models):
            self.add_module(str(idx), model.eval())

        # Store routing info as buffers (not parameters, but will be saved with model)
        # Using register_buffer with persistent=False since these are just routing info
        self.num_models = len(models)
        self.num_external_inputs = num_external_inputs
        self.num_slots = num_slots

        # Store routing as lists of lists (TorchScript compatible)
        self.model_input_indices = model_input_indices
        self.model_output_indices = model_output_indices
        self.final_output_indices = final_output_indices

    def forward(self, *inputs) -> Tuple[torch.Tensor, ...]:
        # Initialize slots with external inputs
        # Using a list for slots - index-based access only
        slots: List[torch.Tensor] = list(inputs)

        # Extend slots to have room for intermediate values
        # We pre-allocate with the first input as placeholder (will be overwritten)
        for _ in range(self.num_slots - self.num_external_inputs):
            slots.append(inputs[0])  # Placeholder, will be overwritten

        # Execute each model in order
        for model_idx in range(self.num_models):
            # Gather inputs by index
            input_indices = self.model_input_indices[model_idx]
            model_inputs = [slots[i] for i in input_indices]

            # Run the model
            model_outputs = self._modules[str(model_idx)](*model_inputs)

            # Normalize to tuple
            if not isinstance(model_outputs, tuple):
                model_outputs = (model_outputs,)

            # Store outputs to their designated slots
            output_indices = self.model_output_indices[model_idx]
            for slot_idx, output_val in zip(output_indices, model_outputs):
                slots[slot_idx] = output_val

        # Gather final outputs
        return tuple(slots[i] for i in self.final_output_indices)


class CombinedNode(LeappNode):
    """A node that combines multiple sequential nodes into a single traced model.

    The routing between nodes is pre-computed as integer indices at construction time,
    making the combined model fully TorchScript-compatible.
    """

    def __init__(self, nodes: List[LeappNode], node_index: int, name: str, backend: str):
        LeappNode.__init__(self, name, node_index)

        # Sort nodes by execution order
        self.nodes = sorted(nodes, key=lambda node: node.node_index)

        # Build external inputs/outputs and routing
        self.inputs, self.outputs, routing = self._build_routing(self.nodes)

        # Extract routing components
        models = [node.compiled_module for node in self.nodes]
        model_input_indices = routing['model_input_indices']
        model_output_indices = routing['model_output_indices']
        final_output_indices = routing['final_output_indices']
        num_slots = routing['num_slots']
        num_external_inputs = len(self.inputs)

        # Create the combined model
        combined_model = CombinedNodeModel(
            models=models,
            num_external_inputs=num_external_inputs,
            model_input_indices=model_input_indices,
            model_output_indices=model_output_indices,
            final_output_indices=final_output_indices,
            num_slots=num_slots
        )

        # Create the backend and compile
        if 'jit' in backend:
            backend = 'jit-trace' # this is more robust. jit script does not have any strengths over trace for this use case

        self.setup_backend(backend, {})
        self.export_backend.override_module_builder(lambda: combined_model)

        self.compile_model()

    def _build_routing(self, nodes: List[LeappNode]) -> Tuple[List, List, Dict]:
        """Build the index-based routing for the combined model.

        Slots are organized as:
        [0..num_external_inputs-1]: External inputs
        [num_external_inputs..]: Intermediate values (outputs from each model)

        Returns:
            Tuple of (external_inputs, external_outputs, routing_dict)
        """
        # Track all tags and their slot indices
        tag_to_slot: Dict[str, int] = {}

        # First pass: identify external inputs (inputs with no internal source)
        # and collect all output tags
        internal_output_tags = set()
        for node in nodes:
            for output in node.outputs:
                if output.tag is not None:
                    internal_output_tags.add(output.tag)

        # External inputs are inputs whose tags don't come from internal outputs
        external_inputs = []
        external_input_tags = OrderedDict()  # tag -> input_description

        for node in nodes:
            for input_desc in node.inputs:
                tag = input_desc.tag
                if tag is None or tag not in internal_output_tags:
                    # This is an external input
                    if tag not in external_input_tags:
                        external_input_tags[tag or input_desc.name] = input_desc
                        external_inputs.append(input_desc)

        # Assign slot indices to external inputs
        for idx, tag in enumerate(external_input_tags.keys()):
            tag_to_slot[tag] = idx

        num_external_inputs = len(external_inputs)
        next_slot = num_external_inputs

        # Second pass: build routing for each model
        model_input_indices = []
        model_output_indices = []

        for node in nodes:
            # Input indices for this model
            input_indices = []
            for input_desc in node.inputs:
                tag = input_desc.tag or input_desc.name
                if tag in tag_to_slot:
                    input_indices.append(tag_to_slot[tag])
                else:
                    raise ValueError(
                        f"Input tag '{tag}' not found in slot mapping for node '{node.name}'")
            model_input_indices.append(input_indices)

            # Output indices for this model - assign new slots
            output_indices = []
            for output_desc in node.outputs:
                tag = output_desc.tag or f"{node.name}/{output_desc.name}"
                tag_to_slot[tag] = next_slot
                output_indices.append(next_slot)
                next_slot += 1
            model_output_indices.append(output_indices)

        # Identify final outputs (outputs from the last node, or outputs not consumed by other nodes)
        # For simplicity, use outputs from the last node that have tags
        last_node = nodes[-1]
        external_outputs = []
        final_output_indices = []

        for output_desc in last_node.outputs:
            tag = output_desc.tag or f"{last_node.name}/{output_desc.name}"
            external_outputs.append(output_desc)
            final_output_indices.append(tag_to_slot[tag])

        routing = {
            'model_input_indices': model_input_indices,
            'model_output_indices': model_output_indices,
            'final_output_indices': final_output_indices,
            'num_slots': next_slot,
        }

        return external_inputs, external_outputs, routing


def get_combined_node(nodes: List[LeappNode], name):
    '''
    build a combined node from a list of nodes. This assumes the following:
    1. All the nodes have the same backend
    2. All nodes can be connected by valid tags
    3. All nodes have undergone i/o reconciliation (input names and output names match)
    '''
    nodes = sorted(nodes, key=lambda node: node.node_index)
    export_backends = set([node.backend for node in nodes])
    node_backends = [node.export_backend.get_backed_model_type()
                     for node in nodes]
    node_backend = set(node_backends)
    if len(node_backend) != 1:
        _get_logger().warning(
            f"skipping combining {name} because not all nodes have the same backend.\n"
            f"got {node_backend}")
        return None
    
    if len(export_backends) != 1:
        _get_logger().warning(
            f"CombinedNode {name} has multiple declared export backends: {export_backends}. \n"
            f"exporting with the default backend for {list(node_backend)[0]}")

    backend = node_backend.pop()
    node_index = nodes[0].node_index
    try:
        combined_node = CombinedNode(
            nodes, node_index, name, backend)
    except Exception as e:
        _get_logger().error(f"Error creating creating merged node {name}: {e}")
        return None
    return combined_node
