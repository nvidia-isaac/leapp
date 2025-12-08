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
    safe_deepcopy,
    get_attribute_value_from_frame,
    tag_data,
    extract_source_from_line_range
)
from leapp._logging import _get_logger
from .leapp_node import LeappNode
import inspect


class FunctionalNode(LeappNode):
    def __init__(self, name, node_index, from_function, backend=None, use_trace=False,
                 backend_params=None, inputs=None, outputs=None, 
                 environment_constants=None, register_buffers=None):
        super().__init__(name, node_index)
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
                f"FunctionalNode '{self.name}': The following names are present in both register_buffers and environment_constants: {overlap}. "
                "Please ensure there is no overlap between these two lists."
            )

        # model settings
        self.setup_backend(backend, backend_params, use_trace)

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

        _get_logger().info(f"Node context initialized: {self.name}")

    def compile_model(self):
        try:
            self.compiled_model = self.export_backend.compile()
        except Exception as e:
            _get_logger().error(f"Error compiling model: {e}")
            raise e

    def compile_trace(self):

        # Extract source code when tracing stops
        self.executed_lines['source_code'], message = extract_source_from_line_range(
            self.executed_lines,
            self.from_function,
            self.name
        )
        if self.executed_lines['source_code'] != "":
            _get_logger().info(message)
        else:
            _get_logger().error(message)

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
                self.add_input(param_name, param_name, param_value)
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
                self.add_output(output_name, output_name, result[i])
        else:
            if not len(return_names) == 1:
                raise Exception(
                    f"Error: {self.name} has {len(return_names)}"
                    " outputs, but only one output is detectd")
            tag_data(result, self.name + '/' + return_names[0] + '/')
            self.add_output(return_names[0], return_names[0], result)

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
                self.add_input(final_input_name, input_name,
                                final_input_value)
        except Exception as e:
            _get_logger().error(f"Error capturing inputs from frame: {e}")
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
                self.add_output(final_output_name,
                                 output_name, final_output_value)
        except Exception as e:
            _get_logger().error(f"Error capturing outputs from frame: {e}")
            raise e
        self.output_frame = frame

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
