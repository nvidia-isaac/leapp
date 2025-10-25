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

from leapp.utils import CompactYamlList
from leapp.enums import MergeCfgEnum
from .graph_gui import visualize_graph
from .leapp_combination_node import get_subgraph_node


class LeappGraph:
    def __init__(self, logger, nodes):
        self.logger = logger
        self.nodes = nodes
        self.node_name_map = {node.name: node.name for node in nodes.values()}

        # process graph connections
        self.logger.section("Processing Node Connections Using Tagged Values")
        self.connections, self.feedback_connections = self._build_connections(
            self.nodes)

        self.logger.section("Reconciling internal i/o names")
        self._reconcile_io_names(self.connections)

        self.logger.section("Discovering graph inputs and outputs")
        self.graph_inputs, self.graph_outputs = self._compile_graph_io(
            self.nodes, self.connections, self.feedback_connections)

    def merge_nodes(self, merge_nodes):
        if merge_nodes != MergeCfgEnum.NO_MERGE:
            self.logger.section(
                f"Merging nodes with the {merge_nodes.value} configuration")

        num_merged = 0

        if merge_nodes == MergeCfgEnum.NO_MERGE:
            return
        elif merge_nodes == MergeCfgEnum.AUTOMATIC:
            merged_node_list, num_merged = self._merge_nodes_automatically()
        elif merge_nodes == MergeCfgEnum.SIGNATURE:
            raise NotImplementedError(
                "Signature merging is not implemented yet")

        # Rediscover connections after merging nodes
        if num_merged:
            self.logger.info(f"Successfully merged {num_merged} nodes")
            self.logger.info("Rediscovering connections after node merge")
            self.connections, self.feedback_connections = self._build_connections(
                self.nodes)
            self.logger.section("Rediscovering graph inputs and outputs")
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
                                 'dangling_inputs': graph_inputs,
                                 'dangling_outputs': graph_outputs}}

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
            duplicates = set([tag for tag in tags if tags.count(tag) > 1])
            if duplicates:
                for duplicate in duplicates:
                    self.logger.error(
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
                        self.logger.info("Found connection: "
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
                        self.logger.info("Found feedback connection: "
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

        self.logger.info(f"Discovered {len(graph_inputs)} graph inputs")
        self.logger.info(f"Discovered {len(graph_outputs)} graph outputs")
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
            self.logger.warning("i/o names changed, this process edits the node specifications, and may produce "
                                "unexpected behavior. Please check the graph for correctness. If this is not desired, "
                                "please make sure to match io names in the source code")
        else:
            self.logger.info("no names changed")

    def _merge_nodes_automatically(self):
        merged = 0
        node_groups = []

        node_groups = []
        connection_candidates = {}
        for connection in self.connections:
            if len(connection['targets']) != 1:
                continue  # this connection is not a simple output to input connection
            source = connection['source']['node']
            target = connection['targets'][0]['node']
            if source not in connection_candidates:
                connection_candidates[source] = []
            elif target is not connection_candidates[source][0]:
                continue  # this connection is not a simple output to input connection
            connection_candidates[source].append(target)

        for source, connections in connection_candidates.items():
            target = connections[0]
            if len(connections) != len(target.inputs) or len(connections) != len(source.outputs):
                continue  # this connection has unaccounted for inputs or outputs

            valid_group = set([source, target])
            for node_group in node_groups:
                if valid_group.intersection(node_group):
                    node_group.update(valid_group)
                    break
            else:
                node_groups.append(valid_group)

        for group in node_groups:
            current_group_sorted = sorted(
                list(group), key=lambda x: x.node_index)
            name = current_group_sorted[0].name
            for node in current_group_sorted[1:]:
                name += "-" + node.name
            self.logger.info("Creating merged node: " + name)
            subgraph_node = get_subgraph_node(
                name=name, nodes=current_group_sorted, logger=self.logger)

            # Remove all nodes in the group from self.nodes
            for node in current_group_sorted:
                self.node_name_map[node.name] = subgraph_node.name
                del self.nodes[node.name]
                merged += 1

            # Insert the subgraph node
            self.nodes[subgraph_node.name] = subgraph_node

        return self.nodes, merged
