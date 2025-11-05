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
import types
import re
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
        # first validate that the input formats detected are valid
        for parameter_description in self.node_context.input_formats:
            if not parameter_description.valid:
                raise ValueError(
                    f"Invalid parameter format for {parameter_description.name_str} "
                    f"when building function header for {self.node_context.name}:"
                    "the parameter likely contains mixed types or nested structures with different types.")
        header_template = "def forward(self, {inputs}) -> {output_types}:\n"
        return_statement_template = "\nreturn {outputs}\n"
        inputs = ", ".join(
            [f"{parameter_description.name_str}: {parameter_description.dtype}" for parameter_description in self.node_context.input_formats])

        # create output types
        output_types = [
            parameter_description.dtype for parameter_description in self.node_context.output_formats]
        if len(output_types) == 0:
            output_types = "None"
        elif len(output_types) == 1:
            output_types = output_types[0]
        else:
            output_types = f"Tuple[{', '.join(output_types)}]"
        header = header_template.format(
            inputs=inputs, output_types=output_types)

        # only include outputs that are declared by user.
        outputs = ", ".join([output.name_raw
                            for output in self.node_context.output_formats
                            if output.name_raw in self.node_context._declared_outputs])
        return_statement = return_statement_template.format(outputs=outputs)

        return header, return_statement, outputs

    def _create_module_template(self):
        # if the function or snippet is a class method, we need to create a module that inherits
        # from the class to preserve class functions
        if 'self' in self.node_context.input_frame.f_locals:
            parent_class = type(self.node_context.input_frame.f_locals['self'])
        else:
            parent_class = None
        module_instance = create_module(
            self.node_context.name, parent_class, self.node_context.cached_constant_values)

        # Set default values as instance attributes (from stored dictionary)
        for default_name, default_value in self.node_context.default_kwargs.items():
            # Set as instance attribute - accessible as self.default_name
            setattr(module_instance, default_name, default_value)
            self.logger.debug(
                f"Set instance attribute (default value): {default_name} = {default_value}")

        # add buffers to the module
        for buffer_name in self.node_context.register_buffers:
            value = self.node_context.cached_buffer_values[buffer_name]
            module_instance.add_buffer(buffer_name, value)

        # copy all the attributes from the original object to the module
        if 'self' in self.node_context.input_frame.f_locals:
            original_obj = self.node_context.input_frame.f_locals['self']
            for attr_name in dir(original_obj):
                # we don't copy any private attributes or register buffers or registered constants
                if attr_name.startswith('__') or attr_name.endswith('__') or \
                        f"self.{attr_name}" in self.node_context.register_buffers or \
                        f"{attr_name}" in module_instance.__constant__:
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

        # Handle inputs where the name was transformed (e.g., "self.var1" -> "self_var1")
        # Inject assignments to map from function parameters back to original names
        for param_format in self.node_context.input_formats:
            original_name = param_format.name_raw  # e.g., "self.var1"
            # e.g., "self_var1" (after transformation)
            param_name = param_format.name_str
            if original_name != param_name:
                # Inject assignment: self.var1 = self_var1
                data_patch_string += f"{original_name} = {param_name}\n"

        return data_patch_string

    def _append_outputs_to_return_statements(self, source_code, outputs_str):
        """
        Append outputs_str to all return statements in the source code.

        For example:
        - 'return val' becomes 'return val, outputs_str'
        - 'return' becomes 'return outputs_str'

        Returns:
            modified_code if return statements were found, False otherwise
        """
        if not outputs_str:
            return source_code, 0

        # Pattern to match return statements
        # This matches 'return' followed by optional whitespace and everything until newline or comment
        # Use list to allow modification in nested function
        replacement_count = [0]

        def replace_return(match):
            replacement_count[0] += 1
            indent = match.group(1)
            return_value = match.group(2).strip()

            if return_value:
                # return has a value, append with comma
                return f"{indent}return {return_value}, {outputs_str}"
            else:
                # return is empty, just add outputs_str
                return f"{indent}return {outputs_str}"

        # Match return statements: capture indentation and the return value
        # Negative lookahead to avoid matching 'return' in strings
        pattern = r'^(\s*)return(\s+[^\n]*)?$'
        modified_code = re.sub(pattern, replace_return,
                               source_code, flags=re.MULTILINE)

        return modified_code, replacement_count[0]

    def create_module_from_source(self):
        m = self._create_module_template()

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

        header, return_statement, outputs_str = self._create_header_and_return_statement()
        function_name = "forward"
        if self.node_context.from_function:
            # For functions, append outputs_str to all return statements
            modified_source, replacement_count = self._append_outputs_to_return_statements(
                source_code, outputs_str)
            source_code = modified_source
            if replacement_count != 0 or len(self.node_context._declared_outputs) == 0:
                return_statement = ""
            # if replacement count is 0, we use the artificial return statement created previously

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
                "NoneType": type(None),
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

    def save(self, save_path: str, compiled_model: torch.jit.ScriptModule) -> Tuple[str, str, str]:
        # Freeze the model before saving for optimization
        if compiled_model is not None:
            preserved_attrs = []
            if hasattr(self, 'node_context') and hasattr(self.node_context, 'saved_buffers'):
                preserved_attrs = self.node_context.saved_buffers

            compiled_model = torch.jit.freeze(
                compiled_model.eval(), preserved_attrs=preserved_attrs)
        else:
            self.logger.error(
                "No compiled model found for {self.node_context.name}")

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
        # input_formats is a list of ParameterFormat objects, resolve each one to get values
        input_values = [resolve_tensor_descriptions_to_values(param_format)
                        for param_format in self.node_context.input_formats]

        compiled_model = torch.jit.trace(
            m, input_values, **self.backend_params)
        # Freezing moved to save() method to allow node combination

        return compiled_model


class TorchScriptExportBackend(TorchExportBackend):
    def compile(self):
        m = self.create_module_from_source()
        compiled_model = torch.jit.script(m, **self.backend_params)
        # Freezing moved to save() method to allow node combination

        return compiled_model
