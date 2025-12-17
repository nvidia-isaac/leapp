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
    safe_deepcopy,
    get_attribute_value_from_frame,
    extract_source_from_line_range
)
from leapp._logging import _get_logger
from leapp.leapp_graph.leapp_node import LeappNode
from leapp.leapp_graph.traced_tensor import TracedTensor


class BlockContextNode(LeappNode):
    def __init__(self, name, node_index, backend=None, use_trace=False,
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
        self.from_function = False
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
                f"BlockContextNode '{self.name}': The following names are present in both register_buffers and environment_constants: {overlap}. "
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


    def compile_model(self):
        try:
            self.compiled_model = self.export_backend.compile()
        except Exception as e:
            _get_logger().error(f"Error compiling model: {e}")
            raise e

    def compile_trace(self, *args):
        # Extract source code when tracing stops
        self.executed_lines['source_code'], message = extract_source_from_line_range(
            self.executed_lines,
            self.name
        )
        if self.executed_lines['source_code'] != "":
            _get_logger().info(message)
        else:
            _get_logger().error(message)

    def create_trace_function(self, skip_file):
        """Create and return the trace function for block context tracing.
        
        Args:
            skip_file: Filename to skip when tracing (e.g., export_manager.py)
        
        Returns:
            A trace function suitable for use with sys.settrace
        """
        def trace_code_snippet(frame, event, arg):
            # Skip tracing the specified file
            code = frame.f_code
            if code.co_filename.split('/')[-1] == skip_file:
                return trace_code_snippet

            # Capture line events to determine the range of executed code
            if event == 'line':
                # Only track lines from the same file as the first line
                if (self.executed_lines['filename'] == code.co_filename and 
                    self.executed_lines['function_name'] == code.co_name):
                    self.executed_lines['lines'].add(frame.f_lineno)
                    self.executed_lines['min_line'] = min(
                        self.executed_lines['min_line'], frame.f_lineno)
                    self.snapshot_buffer_values(frame)
                    self.executed_lines['max_line'] = max(
                        self.executed_lines['max_line'], frame.f_lineno)

            return trace_code_snippet
        return trace_code_snippet
    
    def _check_for_active_traced_tensors(self, data, variable_name, path=None):
        """Recursively check if data contains any TracedTensor instances.
        
        Args:
            data: The data structure to check
            path: Current path in the data structure (for error messages)
            
        Returns:
            tuple: (found, traced_tensor, location) where:
                - found: True if TracedTensor was found
                - traced_tensor: The TracedTensor instance found (or None)
                - location: String describing where it was found
        """
        if not path:
            path = variable_name
        if isinstance(data, TracedTensor) and data.is_tracing:
            _get_logger().error(
                f"Cannot use TracedTensor as input to annotate.block() or annotate.method() '{self.name}'.\n"
                f"Variable '{variable_name}' (at {path}) contains an active TracedTensor"
                f"from node '{data.context}'.\n"
                f"\n"
                f"This happens when you try to use a TracedTensor created by annotate.input_tensors() "
                f"as input to code inside annotate.block().\n"
                f"\n"
                f"You must call annotate.output_tensors() to finalize the TracedTensor node first"
            )
            raise Exception(
                f"Cannot use TracedTensor '{path}' as input to annotate.block() or annotate.method() '{self.name}'. "
                f"Call annotate.output_tensors() first or use .tensor to get the underlying tensor."
            )
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                self._check_for_active_traced_tensors(
                    item, f"{path}[{i}]" if path else f"[{i}]")
        elif isinstance(data, dict):
            for key, value in data.items():
                self._check_for_active_traced_tensors(
                    value, f"{path}['{key}']" if path else f"['{key}']")

    def _capture_specified_value_from_frame(self, variable_name, frame):
        # If variable_name matches *.* pattern, extract from nested objects in frame
        final_variable_name = variable_name
        if "." in variable_name:
            obj, final_variable_name = get_attribute_value_from_frame(
                frame, variable_name)
        else:
            if variable_name in frame.f_locals:
                obj = frame.f_locals[variable_name]
            elif variable_name in frame.f_globals:
                obj = frame.f_globals[variable_name]
            else:
                raise Exception(
                    f"Variable '{variable_name}' not found in frame locals or globals")

        self._check_for_active_traced_tensors(obj, variable_name)
        obj = safe_deepcopy(obj)

        return final_variable_name, safe_deepcopy(obj)

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
        try:
            for output_name in self._declared_outputs:
                obj, _ = get_attribute_value_from_frame(frame, output_name)
                self.tag_data(obj, output_name)

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