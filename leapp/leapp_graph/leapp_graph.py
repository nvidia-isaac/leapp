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
import os
from safetensors.torch import save_file
from collections import Counter

from leapp.utils.tensor_description import CompactYamlList, validate_connection_compatibility
from leapp.utils.logging import _get_logger
from .graph_gui import visualize_graph


class LeappGraph:
    def __init__(self, nodes, graph_name="combined_node"):
        self.nodes = nodes
        self.graph_name = graph_name
        self.node_name_map = {node.name: node.name for node in nodes.values()}

        # process graph connections
        _get_logger().section("Processing Node Connections Using Tagged Values")
        self.connections, self.feedback_connections = self._build_connections(
            self.nodes)

        _get_logger().section("Discovering graph inputs and outputs")
        self.graph_inputs, self.graph_outputs = self._compile_graph_io(
            self.nodes, self.connections, self.feedback_connections)

        _get_logger().section("Validating connection compatibility")
        self._validate_connection_compatibility()

    def get_feedback_initial_values(self):
        feedback_initial_values = {}
        for connection in self.feedback_connections:
            for target in connection['targets']:
                node = target['node']
                input_desc = node.inputs[target['idx']]
                key = f"{node.name}/{input_desc.name_str}"
                value = input_desc.value
                feedback_initial_values[key] = value
        return feedback_initial_values

    def save_feedback_initial_values(self, save_path, name):
        feedback_initial_values = self.get_feedback_initial_values()
        if not feedback_initial_values:
            return

        filename = f"{name}_initial_values.safetensors"
        save_path = os.path.join(save_path, filename)
        _get_logger().info(f"Saving feedback initial values to {save_path}")
        save_file(feedback_initial_values, save_path)
        return filename

    def get_full_pipeline_description(self):
        processed_connections = self._finalize_connections(self.connections)
        processed_feedback_connections = self._finalize_connections(
            self.feedback_connections)
        graph_inputs = {}
        for input in self.graph_inputs:
            if '/' in input:
                node_name, field_name = input.split('/', 1)
                if node_name not in graph_inputs:
                    graph_inputs[node_name] = CompactYamlList()
                graph_inputs[node_name].append(field_name)
        graph_outputs = {}
        for output in self.graph_outputs:
            if '/' in output:
                node_name, field_name = output.split('/', 1)
                if node_name not in graph_outputs:
                    graph_outputs[node_name] = CompactYamlList()
                graph_outputs[node_name].append(field_name)
        pipeline = {'pipeline': {'data_flow': processed_connections,
                                 'feedback_flow': processed_feedback_connections,
                                 'inputs': graph_inputs,
                                 'outputs': graph_outputs}}

        return pipeline

    def visualize(self, save_path, graph_name):
        visualize_graph(self.nodes, self.connections, self.feedback_connections,
                        self.graph_inputs, self.graph_outputs, save_path, graph_name)

    def get_graph_statistics(self):
        internal_connections = 0
        for connection in self.connections:
            internal_connections += len(connection['targets'])
        for connection in self.feedback_connections:
            internal_connections += len(connection['targets'])

        total_edges = internal_connections + \
            len(self.graph_inputs) + len(self.graph_outputs)
        return internal_connections, total_edges

    def _finalize_connections(self, connections_dict):
        processed_connections = {}
        for connection in connections_dict:
            source = connection['source']
            targets = connection['targets']
            source_port = source['node'].name + '/' + \
                source['node'].outputs[source['idx']].name_str
            target_ports = CompactYamlList()
            for target in targets:
                target_ports.append(target['node'].name + '/' +
                                    target['node'].inputs[target['idx']].name_str)

            processed_connections[source_port] = target_ports
        return processed_connections

    def _build_connections(self, nodes):
        connections = {}
        feedback_connections = {}
        for node in nodes.values():
            # first check if any duplicate tags. duplicates are not suppported
            tags = [input.tag for input in node.inputs if input.tag is not None]
            tag_counts = Counter(tags)
            duplicates = {tag for tag, count in tag_counts.items() if count > 1}
            if duplicates:
                duplicate_list = ", ".join(sorted(duplicates))
                _get_logger().fatal(
                    f"Error: unsupported use of sending the same tensor multiple times to the same node. "
                    f"Duplicate input tags in node {node.name}: {duplicate_list}",
                    error_type=Exception)

            for in_idx, input in enumerate(node.inputs):
                if input.tag is None:  # case where the input is dangling
                    pass
                else:
                    source_node_name = input.tag.split('/')[0]

                    source_node = nodes[self.node_name_map[source_node_name]]
                    source_node_output_ports = [
                        output.tag for output in source_node.outputs]
                    if input.tag not in source_node_output_ports:
                        _get_logger().fatal(
                            f"Error: {source_node.name} does not produce tag {input.tag}",
                            error_type=Exception)

                    out_idx = source_node_output_ports.index(input.tag)

                    if source_node.node_index < node.node_index:
                        if input.tag not in connections:
                            connections[input.tag] = {
                                'source': {'node': source_node, 'idx': out_idx},
                                'targets': []
                            }
                        connections[input.tag]['targets'].append(
                            {'node': node, 'idx': in_idx})
                        _get_logger().info("Found connection: "
                                         f"{source_node.name}/{source_node.outputs[out_idx].name_str} "
                                         f" -> {node.name}/{node.inputs[in_idx].name_str}")
                    else:
                        if input.tag not in feedback_connections:
                            feedback_connections[input.tag] = {
                                'source': {'node': source_node, 'idx': out_idx},
                                'targets': []
                            }
                        feedback_connections[input.tag]['targets'].append(
                            {'node': node, 'idx': in_idx})
                        _get_logger().info("Found feedback connection: "
                                         f"{source_node.name}/{source_node.outputs[out_idx].name_str} "
                                         f" -> {node.name}/{node.inputs[in_idx].name_str}")

        return list(connections.values()), list(feedback_connections.values())

    def _compile_graph_io(self, nodes, connections, feedback_connections):
        # any inputs and outputs that are not connected to any nodes are outside connections
        graph_inputs = []
        graph_outputs = []

        # Collect all target ports (destinations) from both forward and feedback connections
        all_target_ports = set()
        for connection in connections:
            for target in connection['targets']:
                target_port = target['node'].name + '/' + \
                    target['node'].inputs[target['idx']].name_str
                all_target_ports.add(target_port)

        for connection in feedback_connections:
            for target in connection['targets']:
                target_port = target['node'].name + '/' + \
                    target['node'].inputs[target['idx']].name_str
                all_target_ports.add(target_port)

        # Collect all source ports from both forward and feedback connections
        all_source_ports = set()
        for connection in connections:
            source = connection['source']
            source_port = source['node'].name + '/' + \
                source['node'].outputs[source['idx']].name_str
            all_source_ports.add(source_port)

        for connection in feedback_connections:
            source = connection['source']
            source_port = source['node'].name + '/' + \
                source['node'].outputs[source['idx']].name_str
            all_source_ports.add(source_port)

        # Find dangling inputs and outputs
        for node in nodes.values():
            # An input is dangling if it's not the target of any internal connection
            for input_desc in node.inputs:
                input_name = input_desc.name_str
                node_input = node.name + '/' + input_name
                if node_input not in all_target_ports:
                    graph_inputs.append(node_input)

            # An output is dangling if it's not the source of any internal connection
            for output_desc in node.outputs:
                output_name = output_desc.name_str
                node_output = node.name + '/' + output_name
                if node_output not in all_source_ports:
                    graph_outputs.append(node_output)

        _get_logger().info(f"Discovered {len(graph_inputs)} graph inputs")
        _get_logger().info(f"Discovered {len(graph_outputs)} graph outputs")
        return graph_inputs, graph_outputs

    def _validate_connection_compatibility(self):
        for connection in self.connections + self.feedback_connections:
            source = connection['source']
            source_desc = source['node'].outputs[source['idx']]
            source_name = f"{source['node'].name}/{source_desc.name_str}"

            for target in connection['targets']:
                target_desc = target['node'].inputs[target['idx']]
                target_name = f"{target['node'].name}/{target_desc.name_str}"
                validate_connection_compatibility(
                    source_name=source_name,
                    source_shape=source_desc.shape,
                    source_dtype=source_desc.dtype,
                    target_name=target_name,
                    target_shape=target_desc.shape,
                    target_dtype=target_desc.dtype,
                )
