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
    extract_source_from_line_range,
    find_with_block_end,
    get_attribute_value_from_namespace,
    frame_to_namespace
)
from leapp._logging import _get_logger
from leapp.leapp_graph.leapp_node import LeappNode
from leapp.leapp_graph.traced_tensor import TracedTensor


class BlockContextNode(LeappNode):
    def __init__(self, name, node_index, backend=None,
                 backend_params=None, inputs=None, outputs=None,
                 environment_constants=None, register_buffers=None):
        super().__init__(name, node_index)
        # input parameters
        # this variable is for temporary use only,
        # all data will be stored in self.inputs after the function is executed
        if inputs is not None:
            # Deduplicate while preserving order (dict.fromkeys preserves insertion order)
            self._declared_inputs = list(dict.fromkeys(inputs))
        else:
            self._declared_inputs = []
        # output parameters
        # this variable is for temporary use only,
        # all data will be stored in self.outputs after the function is executed
        if outputs is not None:
            # Deduplicate while preserving order (dict.fromkeys preserves insertion order)
            self._declared_outputs = list(dict.fromkeys(outputs))
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
        self.setup_backend(backend, backend_params)

        # source code:
        self.executed_lines = {
            'filename': None,
            'function_name': None,
            'min_line': None,
            'max_line': None,
            'lines': set(),
            'source_code': None  # Store extracted source code here
        }

        self.input_namespace = None  # Combined globals + locals from input frame
        self.output_namespace = None  # Combined globals + locals from output frame
        self.cached_buffer_values = {}
        self.cached_constant_values = {}

    def compile_trace(self, *args):
        # Extract source code when tracing stops
        self.executed_lines['source_code'], message = extract_source_from_line_range(
            self.executed_lines,
            self.name,
            is_function=False  # Block contexts use With/AsyncWith AST nodes
        )
        if self.executed_lines['source_code'] != "":
            _get_logger().info(message)
        else:
            _get_logger().error(message)

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
                f"Cannot use TracedTensor as input to block() or method() '{self.name}'.\n"
                f"Variable '{variable_name}' (at {path}) contains an active TracedTensor"
                f"from node '{data.context}'.\n"
                f"\n"
                f"This happens when you try to use a TracedTensor created by input_tensors() "
                f"as input to code inside block().\n"
                f"\n"
                f"You must call output_tensors() to finalize the TracedTensor node first"
            )
            raise Exception(
                f"Cannot use TracedTensor '{path}' as input to block() or method() '{self.name}'. "
                f"Call output_tensors() first or use .tensor to get the underlying tensor."
            )
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                self._check_for_active_traced_tensors(
                    item, f"{path}[{i}]" if path else f"[{i}]")
        elif isinstance(data, dict):
            for key, value in data.items():
                self._check_for_active_traced_tensors(
                    value, f"{path}['{key}']" if path else f"['{key}']")

    def _capture_specified_value_from_namespace(self, variable_name, namespace):
        """Extract a variable value from a namespace dictionary.
        
        Args:
            variable_name: Name of the variable (supports dotted names like 'self.attr')
            namespace: Dictionary containing variable bindings
            
        Returns:
            tuple: (final_variable_name, deepcopied_value)
        """
        obj, final_variable_name = get_attribute_value_from_namespace(
            namespace, variable_name)

        self._check_for_active_traced_tensors(obj, variable_name)
        obj = safe_deepcopy(obj)

        return final_variable_name, safe_deepcopy(obj)

    def capture_inputs_from_namespace(self, namespace):
        """Capture declared inputs from a namespace dictionary.
        
        Args:
            namespace: Dictionary containing variable bindings (globals + locals)
        """
        try:
            for input_name in self._declared_inputs:
                final_input_name, final_input_value = self._capture_specified_value_from_namespace(
                    input_name, namespace)
                self.add_input(final_input_name, input_name,
                               final_input_value)
        except Exception as e:
            _get_logger().error(f"Error capturing inputs from namespace: {e}")
            raise e
        self.input_namespace = namespace

    def capture_outputs_from_namespace(self, namespace):
        """Capture declared outputs from a namespace dictionary.
        
        Args:
            namespace: Dictionary containing variable bindings (globals + locals)
        """
        try:
            for output_name in self._declared_outputs:
                obj, _ = get_attribute_value_from_namespace(namespace, output_name)
                self.tag_data(obj, output_name)

                final_output_name, final_output_value = self._capture_specified_value_from_namespace(
                    output_name, namespace)
                self.add_output(final_output_name,
                                output_name, final_output_value)
        except Exception as e:
            _get_logger().error(f"Error capturing outputs from namespace: {e}")
            raise e
        self.output_namespace = namespace

    def validate_outputs_from_namespace(self, namespace):
        """Validate declared outputs against a namespace dictionary.
        
        Args:
            namespace: Dictionary containing variable bindings (globals + locals)
        """
        try:
            for output_name in self._declared_outputs:
                obj, _ = get_attribute_value_from_namespace(namespace, output_name)
                self.tag_data(obj, output_name)
                final_output_name, final_output_value = self._capture_specified_value_from_namespace(
                    output_name, namespace)
                self.validate_output_and_update_tags(
                    final_output_name, output_name, final_output_value)
        except Exception as e:
            _get_logger().error(f"Error validating outputs from namespace: {e}")
            raise e
        self.output_namespace = namespace

    def validate_inputs_from_namespace(self, namespace):
        """Validate declared inputs against a namespace dictionary.
        
        Args:
            namespace: Dictionary containing variable bindings (globals + locals)
        """
        try:
            for input_name in self._declared_inputs:
                final_input_name, final_input_value = self._capture_specified_value_from_namespace(
                    input_name, namespace)
                self.validate_input_and_update_tags(
                    final_input_name, input_name, final_input_value)
        except Exception as e:
            _get_logger().error(f"Error validating inputs from namespace: {e}")
            raise e
        self.input_namespace = namespace
    
    def validate_function_boundaries(self, caller_frame):
        # Check if this is a re-entry - validate block boundaries match
        new_executed_lines = {
            'filename': caller_frame.f_code.co_filename,
            'function_name': caller_frame.f_code.co_name,
            'min_line': caller_frame.f_lineno,
            'max_line': find_with_block_end(caller_frame.f_code.co_filename, caller_frame.f_lineno)
        }

        for key in ('filename', 'function_name', 'min_line', 'max_line'):
            if self.executed_lines[key] != new_executed_lines[key]:
                _get_logger().error(
                    f"Error: {self.name} re-entered but block boundaries do not match\n"
                    f"Original: {self.executed_lines}\n"
                    f"New: {new_executed_lines}")
                raise Exception(
                    f"Error: {self.name} re-entered but block boundaries do not match")

    def snapshot_buffer_values(self, namespace):
        """Cache buffer and constant values from a namespace.
        
        Args:
            namespace: Dictionary containing variable bindings
        """
        for buffer_name in self.register_buffers:
            value, _ = get_attribute_value_from_namespace(namespace, buffer_name)
            self._cache_buffer_value(buffer_name, value)

        for constant_name in self.environment_constants:
            value, _ = get_attribute_value_from_namespace(namespace, constant_name)
            self._cache_constant_value(constant_name, value)

    def _cache_buffer_value(self, buffer_name, value):
        if buffer_name not in self.cached_buffer_values:
            self.cached_buffer_values[buffer_name] = safe_deepcopy(value)

    def _cache_constant_value(self, constant_name, value):
        if constant_name not in self.cached_constant_values:
            self.cached_constant_values[constant_name] = safe_deepcopy(value)
