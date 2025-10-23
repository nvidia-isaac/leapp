from .graph_element import LeappGraphElement
from collections import OrderedDict
from typing import List
import torch
from leapp.utils import CompactYamlList, reconstruct_from_named_dict, flatten_to_named_dict


class SubgraphNodeModel(torch.nn.Module):
    def __init__(self, name_order: List[str], models, input_names, output_names, io_configs):
        super().__init__()
        self.input_names = input_names
        self.output_names = output_names
        if (any(output_name is None for output_name in output_names)):
            raise ValueError("Unexpected untagged output in subgraph node")
        self.name_order = name_order
        self.io_configs = io_configs
        for idx, model in enumerate(models):
            self.add_module(name_order[idx], model)

    def forward(self, *inputs):
        variable_dict = {}
        # initialize input_names
        for idx, input_name in enumerate(self.input_names):
            variable_dict[input_name] = inputs[idx]

        for idx, name in enumerate(self.name_order):
            config = self.io_configs[name]
            inputs = config['inputs']
            outputs = config['outputs']
            input_formats = config['input_formats']
            output_formats = config['output_formats']

            # extract input values from variable dict
            input_value_dict = {}
            for input_val in inputs:
                if input_val.name in variable_dict:
                    input_value_dict[input_val.name] = variable_dict[input_val.name]
                elif input_val.tag in variable_dict:
                    input_value_dict[input_val.tag] = variable_dict[input_val.tag]
                else:
                    raise ValueError(
                        f"Input {input_val.name} or {input_val.tag} not found in variable_dict")

            # build the input format
            inputs = reconstruct_from_named_dict(
                input_value_dict, input_formats)
            # run the model
            outputs = getattr(self, name)(*inputs)
            # flatten the outputs and commit to variable_dict
            if not isinstance(outputs, tuple):
                # if output formats is a single value, convert for unity
                outputs = [outputs]
            else:
                outputs = list(outputs)
            output_value_dict = flatten_to_named_dict(outputs, output_formats)
            variable_dict.update(output_value_dict)
        outputs = []
        for output_name in self.output_names:
            if output_name not in variable_dict:
                raise ValueError(
                    f"Output {output_name} not found in variable_dict")
            outputs.append(variable_dict[output_name])
        return tuple(outputs)


class SubgraphNodeContext(LeappGraphElement):
    def __init__(self, nodes: List[LeappGraphElement], logger,
                 node_index: int, name, backend):
        nodes_enable_fp16 = [node.enable_fp16 for node in nodes]
        nodes_enable_cuda_graphs = [node.enable_cuda_graphs for node in nodes]
        if not all(node_enable_fp16 == nodes_enable_fp16[0] for node_enable_fp16 in nodes_enable_fp16):
            raise ValueError("All nodes must have the same enable_fp16")
        if not all(node_enable_cuda_graphs == nodes_enable_cuda_graphs[0] for node_enable_cuda_graphs in nodes_enable_cuda_graphs):
            raise ValueError("All nodes must have the same enable_cuda_graphs")

        LeappGraphElement.__init__(
            self, name, node_index, logger, backend, nodes_enable_fp16[0], nodes_enable_cuda_graphs[0])

        self.nodes = sorted(nodes, key=lambda node: node.node_index)
        self.node_execution_order = [node.name for node in nodes]
        self.inputs, self.outputs = self._get_graph_level_io(self.nodes)
        node_configs = self._get_per_node_io_formatting(self.nodes)

        input_names = [input.name for input in self.inputs]
        output_names = [output.tag for output in self.outputs]
        models = [node.get_compiled_model() for node in self.nodes]
        combined_model = SubgraphNodeModel(
            self.node_execution_order, models, input_names, output_names, node_configs)
        input_values = [input_val.value for input_val in self.inputs]

        self.export_backend = self._setup_backend(self.backend, {})

        # TODO: This is a temporary hack to get the combined model working.
        # the real solution is to allow the backend to handle existing models
        if self.get_backend() == 'torch':
            self.compiled_model = torch.jit.trace(
                combined_model, input_values)
            self.model_path = 'combined_model.pt'

    def _get_graph_level_io(self, nodes: List[LeappGraphElement]):
        inputs = OrderedDict()
        outputs = OrderedDict()
        for node in nodes:
            for idx, input_val in enumerate(node.inputs):
                if input_val.tag is not None and input_val.tag in outputs:
                    del outputs[input_val.tag]
                else:
                    if input_val.tag in inputs:
                        raise ValueError(
                            f"Input {input_val.tag} tag values should be unique")
                    tag_name = input_val.tag if input_val.tag is not None else input_val.name_str + \
                        "["+str(idx)+"]"
                    inputs[tag_name] = input_val
            for output_val in node.outputs:
                if output_val.tag is not None:
                    outputs[output_val.tag] = output_val
                else:
                    tag_name = node.name + '/' + output_val.name_str
                    outputs[tag_name] = output_val

        return list(inputs.values()), list(outputs.values())

    def get_description(self):
        description = super().get_description()
        description['formatting']['input_format'] = CompactYamlList(
            [input.name for input in self.inputs])
        if len(self.outputs) == 1:
            description['formatting']['output_format'] = self.outputs[0].name
        else:
            description['formatting']['output_format'] = CompactYamlList(
                [output.name for output in self.outputs])
        return description

    def _get_per_node_io_formatting(self, nodes: List[LeappGraphElement]):
        node_configs = {}
        for node in nodes:
            # format
            node_configs[node.name] = {
                'input_formats': node.input_formats,
                'output_formats': node.output_formats,
                'inputs': node.inputs,
                'outputs': node.outputs
            }

        return node_configs


def get_subgraph_node(nodes: List[LeappGraphElement], logger, name):
    '''
    build a subgraph node from a list of nodes. This assumes the following:
    1. All the nodes have the same backend
    2. All nodes can be connected by valid tags
    3. All nodes have undergone i/o reconciliation (input names and output names match)
    '''
    nodes = sorted(nodes, key=lambda node: node.node_index)
    node_backends = [node.export_backend.get_backed_model_type()
                     for node in nodes]
    if not all(backend == node_backends[0] for backend in node_backends):
        logger.warning(
            f"skipping combining {name} because not all nodes have the same backend")
        return None
    backend = node_backends[0]
    node_index = nodes[0].node_index
    name_order = [node.name for node in nodes]

    subgraph_node = SubgraphNodeContext(
        nodes, name_order, node_index, name, backend)
    return subgraph_node
