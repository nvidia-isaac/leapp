import yaml
import os
import torch
import json
from typing import Dict

from tensordict import TensorDict

from leapp.leapp_graph.leapp_node import LeappNode
from leapp._logging import _get_logger
from leapp.utils import map_to_torch_dtype

from leapp.backends.torch_export_backend import TorchExportBackend
from leapp.backends.onnx_export_backend import ONNXExportBackend

class NodeManager:
    def __init__(self, name, model_path, inputs, outputs, parameters):
        self.name = name
        self.model_path = model_path
        self.input_descriptions = inputs
        self.output_descriptions = outputs

        backend = self._create_backend(parameters['backend'])
        self.model = backend.load(model_path, parameters['sha256sum'], parameters['device'])

        self.device = parameters['device']

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
            input_shape = json.loads(input_val['shape']) if isinstance(input_val['shape'], str) else input_val['shape']
            input_dtype = input_val['dtype']
            input_value = torch.zeros(tuple(input_shape), dtype=map_to_torch_dtype(input_dtype))
            input_values_to_populate[input_name] = input_value.to(self.device)
            
        return input_values_to_populate

    @property
    def mock_output(self):
        output_values_to_populate = {}
        for output_val in self.output_descriptions:
            output_name = output_val['name']
            output_shape = json.loads(output_val['shape']) if isinstance(output_val['shape'], str) else output_val['shape']
            output_dtype = output_val['dtype']
            output_value = torch.zeros(tuple(output_shape), dtype=map_to_torch_dtype(output_dtype))
            output_values_to_populate[output_name] = output_value.to(self.device)
        
        return output_values_to_populate

    def _create_backend(self, backend):
        'utilizes the backends just to load the model. the backend is not used for compilation or saving.'
        if backend == "torch":
            return TorchExportBackend(None)
        elif backend == "onnx":
            return ONNXExportBackend(None)
        else:
            raise ValueError(f"Unsupported backend: {backend}")


    def __call__(self, *inputs):
        if len(inputs) != len(self.input_descriptions):
            _get_logger().error(
                f"Expected {len(self.input_descriptions)} inputs, got {len(inputs)}")
            raise ValueError(
                f"Expected {len(self.input_descriptions)} inputs, got {len(inputs)}")

        outputs = self.model(*inputs)

        if len(self.output_descriptions) == 1 and not isinstance(outputs, tuple):
            return outputs
        elif len(self.output_descriptions) > 1 and len(outputs) == len(self.output_descriptions):
            return tuple(outputs)
        else:
            _get_logger().error(
                f"Expected {len(self.output_descriptions)} outputs, got {len(outputs)}")
            raise ValueError(
                f"Expected {len(self.output_descriptions)} outputs, got {len(outputs)}")


class InferenceManager:
    def __init__(self, model_path, verbose=False):
        _get_logger().configure(verbose=verbose, savepath='.')

        # data reading variables
        self.models = None
        self.pipeline = None
        self.system_info = None

        # runtime variables
        

        if not os.path.exists(model_path):
            _get_logger().error(
                f"Leapp description file not found at {model_path}")
            raise FileNotFoundError(
                f"Leapp description file not found at {model_path}")

        if not model_path.endswith(".yaml"):
            _get_logger().error(
                f"Leapp description file must end with .yaml, got {model_path}")
            raise ValueError(
                f"Leapp description file must end with .yaml, got {model_path}")

        self.model_path = model_path

        self._load_description()

        self.nodes = self._create_nodes()
        _get_logger().info(f"Created {len(self.nodes)} nodes")

        self._validate_and_create_inference_graph()

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
            if any(key not in parameters for key in ["model_path", "md5sum", "sha256sum", "device", "backend"]):
                raise ValueError(
                    f"Model description must contain model_path, sha256sum, device, and backend, got {parameters.keys()}")

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
        self.value_dict = TensorDict({}, batch_size = [])

        # inputs
        input_values_to_populate = {}
        for node_name, node in self.nodes.items():
            input_values_to_populate[node_name] = node.mock_input
        
        output_values_to_populate = {}
        for node_name, node in self.nodes.items():
            output_values_to_populate[node_name] = node.mock_output
        
        self.value_dict.update({'inputs':input_values_to_populate}) #configures keys and prealocates the data
        self.value_dict.update({'outputs':output_values_to_populate})

        self.organized_pipeline_connections = {}
        #organize the pipeline connections
        for source, targets in self.pipeline['data_flow'].items():
            source_node_name, source_output_name = source.split('/')
            if source_node_name not in self.organized_pipeline_connections:
                self.organized_pipeline_connections[source_node_name] = {}
            self.organized_pipeline_connections[source_node_name][source_output_name] = []
            for target in targets:
                target_node_name, target_input_name = target.split('/')
                self.organized_pipeline_connections[source_node_name][source_output_name].append((target_node_name, target_input_name))

        self.return_value_list = self.pipeline['outputs']

        self.value_dict.lock_()
    
    def build_return_value_from_outputs(self):
        return_value = {}
        for node_name, output_names in self.return_value_list.items():
            outputs = {}
            for output_name in output_names:
                outputs[output_name] = self.value_dict['outputs'][node_name][output_name]

            return_value[node_name] = outputs
        return return_value


    def run_policy(self, inputs: Dict[str, Dict[str, torch.Tensor]]):
        # TODO: Validate the all expected inputs are present
        # update will corrupt the data if called within a try/except block. do not call within a try/except block.
        self.value_dict['inputs'].update_(inputs)
        for node_name, node in self.nodes.items():
            print("="*60)
            print("FRANK DEBUG: running", node_name)
            # inference with the model
            outputs = node(*self.value_dict['inputs'][node_name].values())
            output_order = node.output_names
            for output_name, output_value in zip(output_order, outputs):
                print("FRANK DEBUG: sampling", output_name)
                self.value_dict['outputs'][node_name][output_name].copy_(output_value)

            # continue if this is a leaf node
            if node_name not in self.organized_pipeline_connections:
                continue

            # update the input values
            connections = self.organized_pipeline_connections[node_name]
            for output_name, targets in connections.items():
                # import pdb; pdb.set_trace()
                for i in range(len(targets)):
                    tensor_val = self.value_dict['outputs'][node_name][output_name]
                    if i > 0:
                        tensor_val = tensor_val.clone() # clone if this tensor is being used by multiple downstream. This prevents corruption of the data.
                    target_node_name, target_input_name = targets[i]
                    print("FRANK DEBUG: updating", target_node_name, target_input_name)
                    print("FRANK DEBUG: tensor_val", tensor_val)
                    self.value_dict['inputs'][target_node_name][target_input_name].copy_(tensor_val)

        return self.build_return_value_from_outputs()
            
    
    def __call__(self, inputs: Dict[str, Dict[str, torch.Tensor]]):
        return self.run_policy(inputs)
        
