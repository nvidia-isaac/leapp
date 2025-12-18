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
import collections

import torch
from leapp.utils import (CompactYamlList,
                         CompactYamlDict,
                         resolve_tensor_descriptions_to_names,
                         describe_io,
                         tag_tensor,)
from leapp.leapp_graph.traced_tensor import TracedTensor
from leapp.backends.export_backend import NoneExportBackend
from leapp._logging import _get_logger
import functools


class LeappNode():
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only wrap if this class defines its own __init__
        if '__init__' in cls.__dict__:
            original_init = cls.__init__

            @functools.wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                # Only log if this is the actual class being instantiated
                # (not a parent's __init__ being called via super())
                if type(self) is cls:
                    _get_logger().info(
                        f"Node context initialized: {self.name}")
            cls.__init__ = wrapped_init

        # Wrap compile_trace to set _model_captured = True after execution
        if 'compile_trace' in cls.__dict__:
            original_compile_trace = cls.compile_trace

            @functools.wraps(original_compile_trace)
            def wrapped_compile_trace(self, *args, **kwargs):
                result = original_compile_trace(self, *args, **kwargs)
                self._model_captured = True
                return result
            cls.compile_trace = wrapped_compile_trace

    def __init__(self, name, node_index):
        self.name = name
        self.node_index = node_index
        self.enable_fp16 = False  # legacy for holoinfer
        self.enable_cuda_graphs = False  # legacy for holoinfer

        # model settings
        self._model_captured = False
        self.compiled_model = None
        self.model_path = None
        self.md5sum = None
        self.sha256sum = None
        self.model_device = None
        self.export_backend = NoneExportBackend(self, {})

        # i/o settings
        self.inputs = []
        self.outputs = []
        self.input_formats = []
        self.output_formats = []

    @property
    def captured(self):
        # this is defaulted to False. the compile_trace method should set this to true
        return self._model_captured

    def get_description(self):
        # dynamically generate i/o descriptions depending on need
        # Directly use the TensorDescription objects in self.inputs
        input_descriptions = [input.dict() for input in self.inputs]

        # Resolve ParameterFormat objects in input_formats to their string names
        # input_formats is a list of ParameterFormat objects, resolve each one
        resolved_input_formats = [resolve_tensor_descriptions_to_names(param_format)
                                  for param_format in self.input_formats]
        input_formats = CompactYamlList(resolved_input_formats)

        # Directly use the TensorDescription objects in self.outputs
        output_descriptions = [output.dict() for output in self.outputs]

        # Resolve ParameterFormat objects in output_formats to their string names
        # output_formats is a list of ParameterFormat objects, resolve each one
        resolved_output_formats = [resolve_tensor_descriptions_to_names(param_format)
                                   for param_format in self.output_formats]

        # Handle single vs multiple outputs
        if len(resolved_output_formats) == 1:
            output_formats = resolved_output_formats[0]
        else:
            output_formats = resolved_output_formats

        if isinstance(output_formats, list):
            output_formats = CompactYamlList(output_formats)
        elif isinstance(output_formats, dict):
            output_formats = CompactYamlDict(output_formats)

        description = {}
        description['inputs'] = input_descriptions
        description['outputs'] = output_descriptions
        description['parameters'] = {
            'model_path': self.model_path,
            'md5sum': self.md5sum,
            'sha256sum': self.sha256sum,
            'device': self.model_device,
            'is_engine_path': self.is_engine_path(),
            'backend': self.get_backend(),
            'enable_fp16': self.enable_fp16,  # legacy for holoinfer
            'enable_cuda_graphs': self.enable_cuda_graphs,  # legacy for holoinfer
        }
        description['formatting'] = {
            'input_format': input_formats,
            'output_format': output_formats,
        }

        return description

    def setup_backend(self, backend, backend_params, use_trace=False):
        if backend == "torch":
            if use_trace:
                backend = "torch-trace"
            else:
                backend = "torch-script"
        if backend is None:
            self.export_backend = NoneExportBackend(
                self, backend_params)
        elif backend == "torch":
            from leapp.backends.torch_export_backend import TorchExportBackend
            self.export_backend = TorchExportBackend(
                self, backend_params)
        elif backend == "torch-script":
            from leapp.backends.torch_export_backend import TorchScriptExportBackend
            self.export_backend = TorchScriptExportBackend(
                self, backend_params)
        elif backend == "torch-trace":
            from leapp.backends.torch_export_backend import TorchTraceExportBackend
            self.export_backend = TorchTraceExportBackend(
                self, backend_params)
        elif backend == "onnx":
            from leapp.backends.onnx_export_backend import ONNXExportBackend
            self.export_backend = ONNXExportBackend(
                self, backend_params)
        elif backend == "cpp":
            raise Exception("C++ backend not implemented")
        elif backend == "py":
            raise Exception("Python backend not implemented")
        else:
            raise Exception(
                f"{self.name} Unexpected backend: {backend}, \n"
                "please use one of the following: torch, onnx, cpp, py")

    def save_model(self, save_path: str):
        self.model_path, self.md5sum, self.sha256sum = self.export_backend.save(
            save_path, self.compiled_model)
        self.model_device = 'cuda'

    def compile_model(self):
        try:
            self.compiled_model = self.export_backend.compile()
        except Exception as e:
            _get_logger().error(f"Error compiling model {self.name}: {e}")
            raise e

    def get_backend(self):
        return self.export_backend.get_backed_model_type()

    def is_engine_path(self):
        if self.get_backend() == 'trt':
            return True
        return False

    def get_compiled_model(self):
        if self.compiled_model is None:
            raise Exception(
                f"Error: {self.name} has no compiled model, please export the model first")
        return self.compiled_model

    def tag_data(self, tensor, tag):
        # the tag is the name of the tensor, with the node name prepended
        tag = self.name + '/' + tag + '/'

        if isinstance(tensor, torch.Tensor):
            if isinstance(tensor, TracedTensor):
                tensor = tensor.tensor
            tag_tensor(tensor, tag)
        elif isinstance(tensor, collections.abc.Mapping):
            for key, value in tensor.items():
                self.tag_data(value, tag + "[" + key + "]")
        elif isinstance(tensor, collections.abc.Iterable) and not isinstance(tensor, (str, bytes)) and not hasattr(tensor, '__array__'):
            # This catches lists, tuples, sets, etc. but excludes strings, bytes, and numpy arrays
            for idx, item in enumerate(tensor):
                self.tag_data(item, tag + "[" + str(idx) + "]")
        else:
            print(
                f"\033[93mWarning: Untaggable datatype in i/o: {type(tensor)}\033[0m")

    @staticmethod
    def _validate_and_add_to_list(descriptions, current_io_list, node_name):
        existing_names = set([io.name_str for io in current_io_list])
        for description in descriptions:
            if description.name_str in existing_names:
                _get_logger().error(
                    f"Duplicate i/o name '{description.name_str}' in node '{node_name}'. \n"
                    f"Currently existing names: {existing_names}\n"
                    f"Each input/output in the same node must have a unique name."
                )
                raise Exception(
                    f"Duplicate name '{description.name_str}' in node '{node_name}'. "
                    f"Each input/output must have a unique name."
                )
            existing_names.add(description.name_str)
            current_io_list.append(description)

    def add_output(self, outout_name, raw_output_name, output_value):
        io_descriptions, output_format = describe_io(
            outout_name, raw_output_name, output_value)
        self._validate_and_add_to_list(io_descriptions, self.outputs, self.name)
        self.output_formats.append(output_format)

    def add_input(self, input_name, raw_input_name, input_value):
        io_descriptions, input_format = describe_io(
            input_name, raw_input_name, input_value)
        self._validate_and_add_to_list(io_descriptions, self.inputs, self.name)
        self.input_formats.append(input_format)
    
    def validate_io_and_update_tags(self, io_name, raw_io_name, io_value, current_io_list):
        '''
        this is used for rerunning the the tracing. each time we run it we 
        we want to validate that the inputs are consistent with the previous run
        and also update tags for feedback detection.   
        '''
        io_descriptions, input_format = describe_io()
    
    def validate_input_and_update_tags(self, input_name, raw_input_name, input_value):
        self.validate_io_and_update_tags(input_name, raw_input_name, input_value, self.inputs)
    def validate_output_and_update_tags(self, output_name, raw_output_name, output_value):
        self.validate_io_and_update_tags(output_name, raw_output_name, output_value, self.outputs)
    
    def change_input_name(self, old_name, new_name):
        _get_logger().warning(
            f"changing input name from {old_name} to {new_name} for model {self.name}")
        if old_name == new_name:
            return
        current_input_names = [input.name_str for input in self.inputs]
        if new_name in current_input_names:
            raise Exception(
                f"Error requesting input name change for {self.name}/{old_name}:"
                f" {new_name} is already in use")
        for input in self.inputs:
            if input.name_str == old_name:
                input.change_name(new_name)

    def change_output_name(self, old_name, new_name):
        _get_logger().warning(
            f"changing output name from {old_name} to {new_name} for model {self.name}")
        if old_name == new_name:
            return
        current_output_names = [output.name_str for output in self.outputs]
        if new_name in current_output_names:
            raise Exception(
                f"Error requesting output name change for {self.name}:"
                f" {new_name} is already in use")
        for output in self.outputs:
            if output.name_str == old_name:
                output.change_name(new_name)

    def compile_trace(self, *args):
        raise NotImplementedError(
            f"compile_trace not implemented for {self.__class__.__name__}")
