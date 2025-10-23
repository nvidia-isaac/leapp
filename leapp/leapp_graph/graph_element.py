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
from leapp.utils import (CompactYamlList,
                         CompactYamlDict,
                         resolve_tensor_descriptions_to_names)


class LeappGraphElement():
    def __init__(self, name, node_index, logger=None, backend=None,
                 enable_fp16=False, enable_cuda_graphs=False):
        self.name = name
        self.node_index = node_index
        self.logger = logger
        self.enable_fp16 = enable_fp16
        self.enable_cuda_graphs = enable_cuda_graphs

        # model settings
        self.compiled_model = None
        self.model_path = None
        self.md5sum = None
        self.sha256sum = None
        self.model_device = None
        self.backend = backend

        # i/o settings
        self.inputs = []
        self.outputs = []
        self.input_formats = []
        self.output_formats = []

    def get_description(self):
        # dynamically generate i/o descriptions depending on need
        # Directly use the TensorDescription objects in self.inputs
        input_descriptions = [input.dict() for input in self.inputs]
        # Resolve TensorDescription objects in input_formats to their string names
        input_formats = CompactYamlList(
            resolve_tensor_descriptions_to_names(self.input_formats))

        # Directly use the TensorDescription objects in self.outputs
        output_descriptions = [output.dict() for output in self.outputs]
        # Resolve TensorDescription objects in output_formats to their string names
        output_formats = resolve_tensor_descriptions_to_names(
            self.output_formats)
        if len(output_formats) == 1:
            output_formats = output_formats[0]
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
            'enable_fp16': self.enable_fp16,
            'enable_cuda_graphs': self.enable_cuda_graphs,
        }
        description['formatting'] = {
            'input_format': input_formats,
            'output_format': output_formats,
        }

        return description

    def _setup_backend(self, backend, backend_params):
        if self.backend is None:
            raise ValueError(
                "Error initializing graph element: Backend is not set")
        export_backend = None
        if self.backend is None:
            from leapp.backends.export_backend import NoneExportBackend
            export_backend = NoneExportBackend(
                self, self.logger, backend_params)
        elif self.backend == "torch":
            from leapp.backends.torch import TorchExportBackend
            export_backend = TorchExportBackend(
                self, self.logger, backend_params)
        elif self.backend == "torch-script":
            from leapp.backends.torch import TorchScriptExportBackend
            export_backend = TorchScriptExportBackend(
                self, self.logger, backend_params)
        elif self.backend == "torch-trace":
            from leapp.backends.torch import TorchTraceExportBackend
            export_backend = TorchTraceExportBackend(
                self, self.logger, backend_params)
        elif self.backend == "onnx":
            raise Exception("ONNX backend not implemented")
        elif self.backend == "cpp":
            raise Exception("C++ backend not implemented")
        elif self.backend == "py":
            raise Exception("Python backend not implemented")
        else:
            raise Exception(
                f"{self.name} Unexpected backend: {backend}, \n"
                "please use one of the following: torch, onnx, cpp, py")

        return export_backend

    def save_model(self, save_path: str):
        self.model_path, self.md5sum, self.sha256sum = self.export_backend.save(
            save_path, self.compiled_model)
        self.model_device = 'cuda'

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

    def compile_model(self):
        raise NotImplementedError(
            f"compile_model is not available for {self.name}")
