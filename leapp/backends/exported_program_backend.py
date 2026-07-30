#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import Tuple

import torch

from leapp.backends.export_backend import ExportBackend, prepare_tensors_for_export
from leapp.utils.logging import _get_logger


class ExportedProgramExportBackend(ExportBackend):
    """Export node graphs via ``torch.export`` to ``.pt2`` artifacts."""

    def __init__(self, node_context, backend_params=None):
        super().__init__(node_context, backend_params)
        self.exported_program = None

    def get_backend_metadata(self):
        return {"torch_version": str(torch.__version__)}

    def get_backend_model_type(self):
        return "pt2"

    def compile(self, m: torch.nn.Module = None):
        if m is None:
            m = self.module_builder()
        m = m.eval()

        input_values = [
            tensor_desc.value for tensor_desc in self.node_context.inputs]
        input_values = prepare_tensors_for_export(input_values)
        export_args = tuple(input_values)

        export_kwargs = {}
        if "strict" in self.backend_params:
            export_kwargs["strict"] = self.backend_params["strict"]
        if "dynamic_shapes" in self.backend_params:
            export_kwargs["dynamic_shapes"] = self.backend_params["dynamic_shapes"]

        self.exported_program = torch.export.export(
            m, export_args, **export_kwargs)
        self.compiled_model = self.exported_program.module()
        self.compiled_module = m
        return self.exported_program

    def save(self, save_path: str) -> Tuple[str, str, str]:
        if self.exported_program is None:
            _get_logger().fatal(
                f"No exported program found for {self.node_context.name}",
                error_type=RuntimeError)

        path = os.path.join(save_path, f"{self.node_context.name}.pt2")
        torch.export.save(self.exported_program, path)
        md5sum, sha256sum = self._verify_model_location_and_get_hash(path)
        return path, md5sum, sha256sum

    def load(self, model_path: str, sha256sum: str):
        _, actual_sha256sum = self._verify_model_location_and_get_hash(model_path)
        if actual_sha256sum != sha256sum:
            _get_logger().fatal(
                f"SHA256 checksum mismatch for {model_path}: "
                f"expected {sha256sum}, got {actual_sha256sum}",
                error_type=ValueError)

        self.exported_program = torch.export.load(model_path)
        device = self._select_runtime_device()
        self.compiled_model = self.exported_program.module().to(device)
        self.runtime_device = device
        self.compiled_module = None
