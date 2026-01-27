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
import abc
from typing import Tuple, Any
import os
import hashlib
import shutil
from typing import Callable
import functools
import torch

from leapp._logging import _get_logger
from leapp.backends.module_builder import ModuleBuilder


def prepare_tensors_for_export(tensors):
    """
    Prepare tensors for export by cloning them to escape inference mode.

    Tensors created under torch.inference_mode() cannot participate in autograd,
    which causes torch.export.export() (used by dynamo) to fail with:
    "RuntimeError: Inference tensors cannot be saved for backward."

    Cloning creates new tensors that are not marked as inference tensors.

    Args:
        tensors: A sequence of tensors (or other values) to prepare.

    Returns:
        A tuple of prepared tensors (cloned if they were torch.Tensor).
    """
    prepared = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if hasattr(t, 'original_clone'):
                prepared.append(t.original_clone())
            else:
                prepared.append(t.clone())
        else:
            prepared.append(t)
    return tuple(prepared)


class ExportBackend(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only wrap if this class defines its own __init__
        if 'compile' in cls.__dict__:
            original_compile = cls.compile

            @functools.wraps(original_compile)
            def wrapped_compile(self, *args, **kwargs):
                model = original_compile(self, *args, **kwargs)
                if type(self) is cls:
                    pass
                    # TODO: run post-compilation validation if configured to do so

                return model
            cls.compile = wrapped_compile

    def __init__(self, node_context, backend_params=None):
        self.node_context = node_context
        if backend_params is None:
            self.backend_params = {}
        else:
            self.backend_params = backend_params

        self.module_builder = ModuleBuilder(node_context)

    def override_module_builder(self, module_builder: Callable):
        self.module_builder = module_builder

    def _verify_model_location_and_get_hash(self, model_path):
        if not os.path.exists(model_path):
            _get_logger().error(f"Model file not found at {model_path}")
            return None, None
        with open(model_path, 'rb') as f:
            file_data = f.read()
            md5sum = hashlib.md5(file_data).hexdigest()
            sha256sum = hashlib.sha256(file_data).hexdigest()
        return md5sum, sha256sum

    def _copy_model_to_path(self, model_path, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(model_path):
            return None

        # Check if save_path is the same as the directory containing model_path
        model_dir = os.path.dirname(os.path.abspath(model_path))
        save_dir = os.path.abspath(save_path)
        if model_dir == save_dir:
            # No need to copy if already in the same directory
            return model_path

        # Get the filename from model_path
        filename = os.path.basename(model_path)
        # Create the full destination path
        dest_path = os.path.join(save_path, filename)
        # Copy the file
        shutil.copy2(model_path, dest_path)

        return dest_path

    @abc.abstractmethod
    def get_backed_model_type(self):
        raise NotImplementedError

    @abc.abstractmethod
    def compile(self) -> Any:
        '''
        Compiles the model.

        This function should return the compiled model. The resulting compiled model 
        can be used in recombination with other models.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, save_path: str, compiled_model: Any) -> Tuple[str, str, str]:
        '''
        Save the compiled model to the given path

        This function should apply all necessary optimizations
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, model_path: str, sha256sum: str, device: str):
        raise NotImplementedError


class NoneExportBackend(ExportBackend):
    def __call__(self) -> Any:
        return self.compile()

    def compile(self) -> Any:
        if "model_path" not in self.backend_params:
            _get_logger().warning(
                f"No model path provided for {self.node_context.name}")
            _get_logger().warning("if this is intentional, please provide a path to the correct model "
                                  "in the generated yaml file. Otherwise, please manually fill in the backend parameters.")
            return None

    def save(self, save_path: str, compiled_model=None) -> Tuple[str, str, str]:
        if "model_path" not in self.backend_params or self.backend_params['model_path'] is None:
            return None, None, None
        md5sum, sha256sum = self._verify_model_location_and_get_hash(
            self.backend_params['model_path'])
        model_path = self.backend_params['model_path']

        if "copy_original_model" in self.backend_params and self.backend_params['copy_original_model'] is True:
            model_path = self._copy_model_to_path(model_path, save_path)

        return model_path, md5sum, sha256sum

    def load(self, model_path: str, sha256sum: str, device: str):
        raise NotImplementedError

    def get_backed_model_type(self):
        if "model_path" not in self.backend_params:
            return None

        path = self.backend_params['model_path']
        suffix = path.split('.')[-1]
        if suffix == 'pt':
            return "torch"
        elif suffix == 'pt2':
            return "torchscript2"
        elif suffix == 'onnx':
            return "onnx"
        elif suffix == 'cpp' or suffix == "cc":
            return "cpp"
        elif suffix == 'py':
            return "py"
        elif suffix == 'engine' or suffix == 'plan':
            return 'trt'
        else:
            raise Exception(
                f"Unsupported model file suffix: {suffix}")
