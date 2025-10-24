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
import torch
from leapp.backends.export_backend import ExportBackend
import os
import linecache
import textwrap
import ast
import types
from typing import Tuple, List, Dict
from leapp.utils import (
    create_module,
    resolve_tensor_descriptions_to_values,
    extract_source_from_line_range
)


class TorchExportBackend(ExportBackend):
    def get_backed_model_type(self):
        return "torch"

    def _create_header_and_return_statement(self):
        # create header
        header_template = "def forward(self, {inputs}) -> {output_types}:\n"
        return_statement_template = "\nreturn {outputs}\n"
        inputs = ", ".join(
            [f"{input.name_str}: {type(input.value).__name__}" for input in self.node_context.inputs])
        output_types = [
            type(output.value).__name__ for output in self.node_context.outputs]
        if len(output_types) == 0:
            output_types = "None"
        elif len(output_types) == 1:
            output_types = output_types[0]
        else:
            output_types = f"Tuple[{', '.join(output_types)}]"
        header = header_template.format(
            inputs=inputs, output_types=output_types)

        outputs = ", ".join([output.name_str
                            for output in self.node_context.outputs])
        return_statement = return_statement_template.format(outputs=outputs)

        return header, return_statement

    def _create_module_template(self):
        # if the function or snippet is a class method, we need to create a module that inherits
        # from the class to preserve class functions
        if 'self' in self.node_context.input_frame.f_locals:
            parent_class = type(self.node_context.input_frame.f_locals['self'])
        else:
            parent_class = None

        module_instance = create_module(self.node_context.name, parent_class)

        # Set environment constants as instance attributes
        for const_name in self.node_context.environment_constants:
            if "self." in const_name or const_name == "self":
                continue  # special case for any class variables, we already store all of them
            elif const_name in self.node_context.input_frame.f_locals:
                const_value = self.node_context.input_frame.f_locals[const_name]
            elif const_name in self.node_context.input_frame.f_globals:
                const_value = self.node_context.input_frame.f_globals[const_name]
            else:
                raise Exception(
                    f"Error {self.node_context.name}: Environment constant {const_name} not found in input frame")

            # Set as instance attribute - accessible as self.const_name
            setattr(module_instance, const_name, const_value)
            self.logger.debug(
                f"Set instance attribute: {const_name} = {const_value}")

        # Set default values as instance attributes (from stored dictionary)
        for default_name, default_value in self.node_context.default_kwargs.items():
            # Set as instance attribute - accessible as self.default_name
            setattr(module_instance, default_name, default_value)
            self.logger.debug(
                f"Set instance attribute (default value): {default_name} = {default_value}")
        # copy all the attributes from the original object to the module
        if 'self' in self.node_context.input_frame.f_locals:
            original_obj = self.node_context.input_frame.f_locals['self']
            for attr_name in dir(original_obj):
                # we don't copy any private attributes or register buffers
                if attr_name.startswith('__') or attr_name.endswith('__') or \
                        f"self.{attr_name}" in self.node_context.register_buffers:
                    continue
                try:
                    value = getattr(original_obj, attr_name)
                    self.logger.debug(f"Setting attribute: {attr_name}")

                    # Check if it's a property (read-only or not)
                    if hasattr(type(original_obj), attr_name):
                        prop = getattr(type(original_obj), attr_name)
                        if isinstance(prop, property):
                            # Bypass property setter by setting directly in __dict__
                            module_instance.__dict__[attr_name] = value

                    elif isinstance(value, torch.nn.Module):
                        module_instance.add_module(attr_name, value)

                    elif not callable(value):
                        module_instance.__dict__[attr_name] = value
                    elif callable(value):
                        bound_method = types.MethodType(value, module_instance)
                        setattr(module_instance, attr_name, bound_method)
                except Exception as e:
                    self.logger.error(
                        f"Error setting attribute {attr_name}: {e}")

        return module_instance

    def _create_data_patching_string(self):
        data_patch_template = "{const_name} = self.{const_name}\n"
        data_patch_string = ""
        targets = self.node_context.environment_constants | self.node_context.register_buffers | set(
            self.node_context.default_kwargs.keys())

        for const_name in targets:
            if "self." in const_name or const_name == "self":
                continue  # special case for any class variables, no rename required
            data_patch_string += data_patch_template.format(
                const_name=const_name)
        return data_patch_string

    def _register_buffers_to_module(self, m):
        for buffer_name in self.node_context.register_buffers:
            value = self.node_context.cached_buffer_values[buffer_name]
            m.add_buffer(buffer_name, value)

    def create_module_from_source(self):
        m = self._create_module_template()

        self._register_buffers_to_module(m)

        if self.node_context.input_frame is None or self.node_context.output_frame is None:
            raise Exception(
                f"Input or output frame not found for {self.node_context.name}")

        source_code, message = extract_source_from_line_range(
            self.node_context.executed_lines,
            self.node_context.from_function,
            self.node_context.name
        )

        if source_code == "":
            self.logger.error(message)
            raise Exception(
                f"No source code found for {self.node_context.name}")
        else:
            self.logger.info(message)

        header, return_statement = self._create_header_and_return_statement()
        if self.node_context.from_function:
            return_statement = ""
        function_name = "forward"
        environment_constant_string = self._create_data_patching_string()

        # create function string
        function_body = environment_constant_string + source_code + return_statement

        # Add 4-space indentation to all lines of function body
        indented_function_body = textwrap.indent(function_body, '    ')

        # add the headder
        function_string = header + indented_function_body
        filename = f"generated_{self.node_context.name}_function"

        try:
            self.logger.debug("\n"+function_string)
            code = compile(function_string, filename, "exec")

            # recreate the environment of the function
            namespace = {
                'Tensor': torch.Tensor,
                "Tuple": Tuple,
                "List": List,
                "Dict": Dict,
            }
            if self.node_context.input_frame is not None:
                namespace.update(self.node_context.input_frame.f_locals)
                namespace.update(self.node_context.input_frame.f_globals)

            exec(code, namespace)
        except Exception as e:
            self.logger.error(f"Error compiling function: {e}")
            self.logger.error(function_string)
            raise e

        forward = namespace[function_name]

        lines = [line + '\n' for line in function_string.splitlines()]
        linecache.cache[filename] = (
            len(function_string), None, lines, filename)
        m.forward = types.MethodType(forward, m)
        if len(m.saved_buffers) > 0:
            self.logger.info("Created the following buffers:")
            for buffer_name in m.saved_buffers:
                self.logger.info(
                    f"  - self.{buffer_name}: intialized as {getattr(m, buffer_name)}")
        return m

    def save(self, save_path: str, compiled_model: torch.jit.ScriptModule = None) -> Tuple[str, str, str]:
        if compiled_model is None:
            compiled_model = self.compiled_model
        path = os.path.join(save_path, f"{self.node_context.name}.pt")
        compiled_model.save(path)
        md5sum, sha256sum = self._verify_model_location_and_get_hash(path)
        return path, md5sum, sha256sum

    def compile(self) -> torch.jit.ScriptModule:
        raise NotImplementedError(
            "TorchExportBackend does not support compilation")
        return None


class TorchTraceExportBackend(TorchExportBackend):
    def compile(self):
        if not len(self.node_context.register_buffers) == 0:
            raise Exception(
                "TorchTraceExportBackend does not support buffers, "
                "consider using export_with='torch' without use_trace=True")
        m = self.create_module_from_source()
        inputs = resolve_tensor_descriptions_to_values(
            self.node_context.input_formats)

        self.compiled_model = torch.jit.trace(m, inputs, **self.backend_params)
        if isinstance(m, torch.nn.Module) and hasattr(m, 'forward'):
            torch.jit.freeze(self.compiled_model.eval(),
                             preserved_attrs=m.saved_buffers)

        return self.compiled_model


class TorchScriptExportBackend(TorchExportBackend):
    def compile(self):
        m = self.create_module_from_source()
        self.compiled_model = torch.jit.script(m, **self.backend_params)

        if isinstance(m, torch.nn.Module) and hasattr(m, 'forward'):
            torch.jit.freeze(self.compiled_model.eval(),
                             preserved_attrs=m.saved_buffers)

        return self.compiled_model
