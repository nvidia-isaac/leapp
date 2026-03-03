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
import os
from safetensors.torch import save_file
from collections import Counter

from leapp.utils.tensor_description import CompactYamlList
from leapp.utils.enums import MergeCfgEnum
from leapp._logging import _get_logger
from .graph_gui import visualize_graph
from .combined_node import get_combined_node


class LeappGraph:
    def __init__(self, nodes, graph_name="combined_node"):
        self.nodes = nodes
        self.graph_name = graph_name
        self.node_name_map = {node.name: node.name for node in nodes.values()}

        # process graph connections
        _get_logger().section("Processing Node Connections Using Tagged Values")
        self.connections, self.feedback_connections = self._build_connections(
            self.nodes)

        _get_logger().section("Reconciling internal i/o names")
        self._reconcile_io_names(self.connections)

        _get_logger().section("Discovering graph inputs and outputs")
        self.graph_inputs, self.graph_outputs = self._compile_graph_io(
            self.nodes, self.connections, self.feedback_connections)

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



    def merge_nodes(self, merge_nodes):
        if merge_nodes != MergeCfgEnum.NO_MERGE:
            _get_logger().section(
                f"Merging nodes with the {merge_nodes.value} configuration")

        num_merged = 0

        if merge_nodes == MergeCfgEnum.NO_MERGE:
            return
        elif merge_nodes == MergeCfgEnum.ALL:
            merged_node_list, num_merged = self._merge_nodes_all()
        elif merge_nodes == MergeCfgEnum.AUTOMATIC:
            merged_node_list, num_merged = self._merge_nodes_automatically()
        elif merge_nodes == MergeCfgEnum.SIGNATURE:
            raise NotImplementedError(
                "Signature merging is not implemented yet")

        # Rediscover connections after merging nodes
        if num_merged:
            _get_logger().info(f"Successfully merged {num_merged} nodes")
            _get_logger().info("Rediscovering connections after node merge")
            self.connections, self.feedback_connections = self._build_connections(
                self.nodes)
            _get_logger().section("Rediscovering graph inputs and outputs")
            self.graph_inputs, self.graph_outputs = self._compile_graph_io(
                self.nodes, self.connections, self.feedback_connections)

        return merged_node_list

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
                for duplicate in duplicates:
                    _get_logger().error(
                        f"Found duplicate input with the tag {duplicate} in node {node.name}")
                raise Exception(
                    "Error: unsupported use of sending the same tensor multiple times to the same node")

            for in_idx, input in enumerate(node.inputs):
                if input.tag is None:  # case where the input is dangling
                    pass
                else:
                    source_node_name = input.tag.split('/')[0]

                    source_node = nodes[self.node_name_map[source_node_name]]
                    source_node_output_ports = [
                        output.tag for output in source_node.outputs]
                    if input.tag not in source_node_output_ports:
                        raise Exception(
                            f"Error: {source_node.name} does not produce tag {input.tag}")

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

    def _reconcile_io_names(self, connections):
        names_changed = False
        for connection in connections:
            source = connection['source']
            targets = connection['targets']

            # Use name_str property for TensorDescription objects
            target_names = [target['node'].inputs[target['idx']].name_str
                            for target in targets]
            desired_target_name = target_names[0]
            if not all(name == desired_target_name for name in target_names):
                names_changed = True
                for target in targets:
                    target['node'].change_input_name(
                        target['node'].inputs[target['idx']].name_str, desired_target_name)

            if not source['node'].outputs[source['idx']].name_str == desired_target_name:
                names_changed = True
                source['node'].change_output_name(
                    source['node'].outputs[source['idx']].name_str, desired_target_name)
        if names_changed:
            _get_logger().debug("i/o names changed, this process edits the node specifications, and may produce "
                                "unexpected behavior. Please check the graph for correctness. If this is not desired, "
                                "please make sure to match io names in the source code")
        else:
            _get_logger().debug("no names changed")

    def _merge_nodes_automatically(self):
        merged = 0

        # TODO: feedback connections should be allowed. the simply get converted to register buffers
        # Build set of nodes involved in feedback connections
        feedback_nodes = set()
        for feedback_connection in self.feedback_connections:
            feedback_nodes.add(feedback_connection['source']['node'])
            for target in feedback_connection['targets']:
                feedback_nodes.add(target['node'])

        # first consolidate into a port agnostic graph
        simplified_connections = {}
        for connection in self.connections:
            source = connection['source']['node']
            targets = connection['targets']
            if source not in simplified_connections:
                simplified_connections[source] = set()
            for target in targets:
                simplified_connections[source].add(target['node'])

        # build a representation of the in and out degrees of each node
        # we are explicitly looking for chains of nodes that only connect to a single other node
        in_degree = {node: 0 for node in self.nodes.values()}
        out_degree = {node: 0 for node in self.nodes.values()}
        for source, targets in simplified_connections.items():
            out_degree[source] = len(targets)
            for target in targets:
                in_degree[target] += 1
        possible_sources = set(
            [node for node in self.nodes.values() if out_degree[node] == 1])
        possible_targets = set(
            [node for node in self.nodes.values() if in_degree[node] == 1])

        connection_candidates = []
        for source in possible_sources:
            target_node = list(simplified_connections[source])[0]
            if target_node not in possible_targets:
                continue
            if source.get_backend() != target_node.get_backend():
                continue
            if source in feedback_nodes or target_node in feedback_nodes:
                continue
            connection_candidates.append(set([source, target_node]))

        # join the connection candidates into groups if they can be chained together
        node_groups = []
        for connection_candidate in connection_candidates:
            # Find all groups that intersect with this connection candidate
            matching_groups = []
            non_matching_groups = []  # preserves other chains

            for node_group in node_groups:
                if connection_candidate.intersection(node_group):
                    matching_groups.append(node_group)
                else:
                    non_matching_groups.append(node_group)

            # Merge all matching groups together with the new connection candidate
            if matching_groups:
                merged_group = connection_candidate.union(*matching_groups)
                node_groups = non_matching_groups
                node_groups.append(merged_group)
            else:
                # No matches, add as a new group
                node_groups.append(connection_candidate)

        for group in node_groups:
            current_group_sorted = sorted(
                list(group), key=lambda x: x.node_index)
            name = current_group_sorted[0].name
            for node in current_group_sorted[1:]:
                name += "-" + node.name
            _get_logger().info("Creating merged node: " + name)
            try:
                combined_node = get_combined_node(
                    name=name, nodes=current_group_sorted)
            except Exception as e:
                _get_logger().error(
                    f"Unexpected error creating merged node {name}: {e}")
                _get_logger().error(f"Skipping node merge for {name}")
                continue

            if combined_node is not None:
                # Remove all nodes in the group from self.nodes
                for node in current_group_sorted:
                    _get_logger().debug(
                        f"Removing node {node.name} from nodes dictionary, current existing nodes: {list(self.nodes.keys())}")
                    self.node_name_map[node.name] = combined_node.name
                    del self.nodes[node.name]
                    merged += 1

                # Insert the combined node
                self.nodes[combined_node.name] = combined_node

        return self.nodes, merged
    
    def _merge_nodes_all(self):
        merged = len(self.nodes)

        combined_node = get_combined_node(name=self.graph_name, nodes=list(self.nodes.values()))

        return self.nodes, merged
