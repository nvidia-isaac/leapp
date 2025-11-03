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

from leapp.utils import (
    extract_return_names,
    describe_io,
    safe_deepcopy,
    get_attribute_value_from_frame,
    tag_data,
    extract_source_from_line_range
)
from .graph_element import LeappGraphElement
import inspect


class NodeContext(LeappGraphElement):
    def __init__(self, name, node_index, from_function, logger=None, backend=None, use_trace=False,
                 backend_params=None, inputs=None, outputs=None, environment_constants=None,
                 register_buffers=None, enable_fp16=False, enable_cuda_graphs=False):
        super().__init__(name, node_index, logger, backend)
        # input parameters
        # this variable is for temporary use only,
        # all data will be stored in self.inputs after the function is executed
        if inputs is not None:
            self._declared_inputs = inputs
        else:
            self._declared_inputs = []
        # output parameters
        # this variable is for temporary use only,
        # all data will be stored in self.outputs after the function is executed
        if outputs is not None:
            self._declared_outputs = outputs
        else:
            self._declared_outputs = []

        # node settings
        self.from_function = from_function
        if environment_constants is not None:
            self.environment_constants = set(environment_constants)
        else:
            self.environment_constants = set()
        if register_buffers is not None:
            self.register_buffers = set(register_buffers)
        else:
            self.register_buffers = set()
        self.default_kwargs = {}

        # Check for overlap between register_buffers and environment_constants
        overlap = self.register_buffers & self.environment_constants
        if overlap:
            raise ValueError(
                f"NodeContext '{self.name}': The following names are present in both register_buffers and environment_constants: {overlap}. "
                "Please ensure there is no overlap between these two lists."
            )

        # model settings
        # overwrite backend depending on using trace
        if self.backend == "torch":
            if use_trace:
                self.backend = "torch-trace"
            else:
                self.backend = "torch-script"
        self.export_backend = self._setup_backend(self.backend, backend_params)

        # source code:
        self.executed_lines = {
            'filename': None,
            'function_name': None,
            'min_line': None,
            'max_line': None,
            'lines': set(),
            'source_code': None  # Store extracted source code here
        }
        self.input_frame = None
        self.output_frame = None
        self.cached_buffer_values = {}
        self.cached_constant_values = {}

        self.logger.info(f"Node context initialized: {self.name}")

    def compile_model(self):
        try:
            self.compiled_model = self.export_backend.compile()
        except Exception as e:
            self.logger.error(f"Error compiling model: {e}")
            raise e

    def compile_trace(self):

        # Extract source code when tracing stops
        self.executed_lines['source_code'], message = extract_source_from_line_range(
            self.executed_lines,
            self.from_function,
            self.name
        )
        if self.executed_lines['source_code'] != "":
            self.logger.info(message)
        else:
            self.logger.error(message)

    def inspect_function_inputs(self, func, args, kwargs):
        # Get parameter names from function signature
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        # bind
        bound_args = sig.bind(*args, **kwargs)
        # apply default values
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            if param_name == 'self':
                continue
            # Check if this parameter was explicitly provided or is using default
            param = sig.parameters[param_name]
            was_provided = (
                param_name in kwargs or
                param_names.index(param_name) < len(args)
            )
            if was_provided:
                self._add_input(param_name, param_name, param_value)
            else:
                # this parameter uses the default value in the function header
                self.default_kwargs[param_name] = param_value

    def inspect_function_outputs(self, func, result):
        # Store outputs using actual variable names from return statement
        if result is None:
            return  # no outputs to capture in function
        return_names = extract_return_names(func)

        if isinstance(result, tuple) and len(return_names) == len(result):
            for i in range(len(result)):
                tag_data(result[i], self.name + '/' + return_names[i] + '/')
                if i < len(return_names):
                    output_name = return_names[i]
                else:
                    output_name = f"output{i+1}"
                self._add_output(output_name, output_name, result[i])
        else:
            if not len(return_names) == 1:
                raise Exception(
                    f"Error: {self.name} has {len(return_names)}"
                    " outputs, but only one output is detectd")
            tag_data(result, self.name + '/' + return_names[0] + '/')
            self._add_output(return_names[0], return_names[0], result)

        # extract custom returns from the environment variables

    def _capture_specified_value_from_frame(self, variable_name, frame):
        # If variable_name matches *.* pattern, extract from nested objects in frame
        obj = None
        final_variable_name = variable_name
        if "." in variable_name:
            obj, final_variable_name = get_attribute_value_from_frame(
                frame, variable_name)
            obj = safe_deepcopy(obj)
        else:
            if variable_name in frame.f_locals:
                obj = safe_deepcopy(frame.f_locals[variable_name])
            elif variable_name in frame.f_globals:
                obj = safe_deepcopy(frame.f_globals[variable_name])
            else:
                raise Exception(
                    f"Variable '{variable_name}' not found in frame locals or globals")

        return final_variable_name, obj

    def capture_inputs_from_frame(self, frame):
        try:
            for input_name in self._declared_inputs:
                final_input_name, final_input_value = self._capture_specified_value_from_frame(
                    input_name, frame)
                self._add_input(final_input_name, input_name,
                                final_input_value)
        except Exception as e:
            self.logger.error(f"Error capturing inputs from frame: {e}")
            raise e
        self.input_frame = frame

    def capture_outputs_from_frame(self, frame):
        for output_name in self._declared_outputs:
            obj, _ = get_attribute_value_from_frame(frame, output_name)
            output_tag = self.name + '/' + output_name + '/'
            tag_data(obj, output_tag)

        try:
            for output_name in self._declared_outputs:
                final_output_name, final_output_value = self._capture_specified_value_from_frame(
                    output_name, frame)
                self._add_output(final_output_name,
                                 output_name, final_output_value)
        except Exception as e:
            self.logger.error(f"Error capturing outputs from frame: {e}")
            raise e
        self.output_frame = frame

    def _add_output(self, outout_name, raw_output_name, output_value):
        io_descriptions, output_format = describe_io(
            outout_name, raw_output_name, output_value)
        self.outputs.extend(io_descriptions)
        self.output_formats.append(output_format)

    def _add_input(self, input_name, raw_input_name, input_value):
        io_descriptions, input_format = describe_io(
            input_name, raw_input_name, input_value)
        self.inputs.extend(io_descriptions)
        self.input_formats.append(input_format)

    def snapshot_buffer_values(self, frame):
        for buffer_name in self.register_buffers:
            value, _ = get_attribute_value_from_frame(
                frame, buffer_name)
            self._cache_buffer_value(buffer_name, value)

        for constant_name in self.environment_constants:
            value, _ = get_attribute_value_from_frame(
                frame, constant_name)
            self._cache_constant_value(constant_name, value)

    def _cache_buffer_value(self, buffer_name, value):
        if buffer_name not in self.cached_buffer_values:
            self.cached_buffer_values[buffer_name] = safe_deepcopy(value)

    def _cache_constant_value(self, constant_name, value):
        if constant_name not in self.cached_constant_values:
            self.cached_constant_values[constant_name] = safe_deepcopy(value)

    def change_input_name(self, old_name, new_name):
        self.logger.warning(
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
        self.logger.warning(
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
