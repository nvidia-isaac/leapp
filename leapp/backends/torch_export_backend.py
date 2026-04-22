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
import warnings
from typing import Tuple

import torch
from leapp.backends.export_backend import ExportBackend, prepare_tensors_for_export
from leapp.utils.logging import _get_logger


class TorchExportBackend(ExportBackend):
    def get_backend_metadata(self):
        return {}
    
    def get_backend_model_type(self):
        return "jit"

    def save(self, save_path: str) -> Tuple[str, str, str]:
        compiled_model = self.compiled_model
        # Freeze the model before saving for optimization
        if compiled_model is not None:
            preserved_attrs = []
            if hasattr(self, 'node_context') and hasattr(self.node_context, 'saved_buffers'):
                preserved_attrs = self.node_context.saved_buffers

            compiled_model = torch.jit.freeze(
                compiled_model.eval(), preserved_attrs=preserved_attrs)
        else:
            _get_logger().error(
                f"No compiled model found for {self.node_context.name}")

        path = os.path.join(save_path, f"{self.node_context.name}.pt")
        compiled_model.save(path)
        md5sum, sha256sum = self._verify_model_location_and_get_hash(path)
        return path, md5sum, sha256sum

    def load(self, model_path: str, sha256sum: str):
        self._load_torchscript(model_path, sha256sum)

    def compile(self):
        raise NotImplementedError(
            "TorchExportBackend does not support compilation, please use torch-script or torch-trace instead")
        return None


class TorchTraceExportBackend(TorchExportBackend):
    def compile(self, m: torch.nn.Module = None):
        if not len(self.node_context.register_buffers) == 0:
            raise Exception(
                "TorchTraceExportBackend does not support buffers, "
                "consider using export_with='jit-trace'")
        if m is None:
            m = self.module_builder()
        m = m.eval()
        # Get flat tensor values directly from inputs (not input_formats which preserves nested structure)
        input_values = [
            tensor_desc.value for tensor_desc in self.node_context.inputs]
        # Clone tensors to escape inference mode (inference tensors can't participate in autograd)
        input_values = prepare_tensors_for_export(input_values)

        self.compiled_model = torch.jit.trace(
            m, input_values, **self.backend_params)
        self.compiled_module = m
        # Freezing moved to save() method to allow node combination


class TorchScriptExportBackend(TorchExportBackend):
    def compile(self, m: torch.nn.Module = None):
        if m is None:
            m = self.module_builder()
        m = m.eval()
        torch.jit._state.enable()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "The TorchScript type system doesn't support instance-level "
                    "annotations on empty non-base types in `__init__`.*"
                ),
                category=UserWarning,
            )
            compiled_model = torch.jit.script(m, **self.backend_params)
        self.compiled_model = compiled_model
        self.compiled_module = m
        # Freezing moved to save() method to allow node combination

        return compiled_model
