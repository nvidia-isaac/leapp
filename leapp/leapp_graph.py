from .utils import CompactYamlList


class LeappGraph:
    def __init__(self, logger, nodes):
        self.logger = logger
        self.nodes = nodes

        # process graph connections
        connections = self._build_connections()

        self._reconcile_io_names(connections)

        self.processed_connections = {}
        for connection in connections.values():
            source = connection['source']
            targets = connection['targets']
            source_port = source['node'].name + '/' + \
                source['node'].outputs[source['idx']].name_str
            target_ports = CompactYamlList()
            for target in targets:
                target_ports.append(target['node'].name + '/' +
                                    target['node'].inputs[target['idx']].name_str)

            self.processed_connections[source_port] = target_ports

        self.graph_inputs, self.graph_outputs = self._compile_graph_io(
            self.processed_connections)

    def get_graph_description(self):
        return self.processed_connections, self.graph_inputs, self.graph_outputs

    def _build_connections(self):
        self.logger.section("Processing Node Connections Using Tagged Values")
        connections = {}
        for node in self.nodes.values():
            # first check if any duplicate tags. duplicates are not suppported
            tags = [input.tag for input in node.inputs if input.tag is not None]
            duplicates = set([tag for tag in tags if tags.count(tag) > 1])
            if duplicates:
                for duplicate in duplicates:
                    self.logger.info(
                        f"found duplicate input with the tag {duplicate} in node {node.name}")
                raise Exception(
                    "Error: unsupported use of sending the same tensor multiple times to the same node")

            for in_idx, input in enumerate(node.inputs):
                if input.tag is None:  # case where the input is dangling
                    pass
                else:
                    source_node_name = input.tag.split('/')[0]
                    self.logger.info(f"source node name: {source_node_name}")

                    source_node = self.nodes[source_node_name]
                    source_node_output_ports = [
                        output.tag for output in source_node.outputs]
                    if input.tag not in source_node_output_ports:
                        raise Exception(
                            f"Error: {source_node_name} does not produce tag {input.tag}")

                    out_idx = source_node_output_ports.index(input.tag)

                    if input.tag not in connections:
                        connections[input.tag] = {
                            'source': {'node': source_node, 'idx': out_idx},
                            'targets': []
                        }
                    connections[input.tag]['targets'].append(
                        {'node': node, 'idx': in_idx})

        return connections

    def _compile_graph_io(self, connections):
        # any inputs and outputs that are not connected to any nodes are outside connections
        graph_inputs = []
        graph_outputs = []

        self.logger.section("Discovering graph inputs and outputs")

        # Collect all target connections (destinations)
        all_targets = []
        for targets_list in connections.values():
            all_targets.extend(targets_list)

        for node in self.nodes.values():
            # An input is dangling if it's not the target of any internal connection
            for input_desc in node.inputs:
                input_name = input_desc.name_str  # Use name_str property
                node_input = node.name + '/' + input_name
                if node_input not in all_targets:
                    graph_inputs.append(node_input)

            # An output is dangling if it's not the source of any internal connection
            for output_desc in node.outputs:
                output_name = output_desc.name_str  # Use name_str property
                node_output = node.name + '/' + output_name
                if node_output not in connections:
                    graph_outputs.append(node_output)

        self.logger.info(f"Discovered {len(graph_inputs)} graph inputs")
        self.logger.info(f"Discovered {len(graph_outputs)} graph outputs")
        return graph_inputs, graph_outputs

    def _reconcile_io_names(self, connections):
        names_changed = True
        self.logger.section("Reconciling internal i/o names")
        for connection in connections.values():
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
            self.logger.warning("i/o names changed, this process edits the node specifications, and may produce\n"
                                "unexpected behavior. Please check the graph for correctness. If this is not desired,\n"
                                "please make sure to match io names in the source code")
        else:
            self.logger.info("no names changed")
