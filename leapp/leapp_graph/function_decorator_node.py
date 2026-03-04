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

import inspect

from leapp.utils.utils import (
    safe_deepcopy,
    extract_return_names,
    extract_source_from_line_range,
    frame_to_namespace,
    get_attribute_value_from_namespace,
)
from leapp._logging import _get_logger
from leapp.leapp_graph.leapp_node import LeappNode
from leapp.leapp_graph.datatypes import TracedTensor


class FunctionDecoratorNode(LeappNode):
    """Node type for the legacy sys.settrace-based _method() decorator.

    Captures function source code, inputs/outputs from namespaces, and builds
    an nn.Module via ModuleBuilder for export.
    """

    def __init__(self, name, node_index, backend=None,
                 backend_params=None, inputs=None, outputs=None,
                 environment_constants=None, register_buffers=None, dry_run=False):
        super().__init__(name, node_index, dry_run=dry_run)

        if inputs is not None:
            self._declared_inputs = list(dict.fromkeys(inputs))
        else:
            self._declared_inputs = []
        if outputs is not None:
            self._declared_outputs = list(dict.fromkeys(outputs))
        else:
            self._declared_outputs = []

        self.from_function = True

        if environment_constants is not None:
            self.environment_constants = set(environment_constants)
        else:
            self.environment_constants = set()
        if register_buffers is not None:
            self.register_buffers = set(register_buffers)
        else:
            self.register_buffers = set()
        self.default_kwargs = {}

        overlap = self.register_buffers & self.environment_constants
        if overlap:
            raise ValueError(
                f"FunctionDecoratorNode '{self.name}': The following names are "
                f"present in both register_buffers and environment_constants: {overlap}. "
                "Please ensure there is no overlap between these two lists."
            )

        self.setup_backend(backend, backend_params)

        self.executed_lines = {
            'filename': None,
            'function_name': None,
            'min_line': None,
            'max_line': None,
            'lines': set(),
            'source_code': None,
        }

        self.input_namespace = None
        self.output_namespace = None
        self.cached_buffer_values = {}
        self.cached_constant_values = {}

    # ── compile / trace ──────────────────────────────────────────────────

    def compile_trace(self, func, *args):
        unwrapped_func = inspect.unwrap(func)

        if unwrapped_func is not func:
            _get_logger().warning(
                f"Function '{func.__name__}' is wrapped by another decorator (e.g., @torch.no_grad()). "
                f"Tracing will only capture the inner function's code, not the wrapper's behavior."
            )

        func_code = unwrapped_func.__code__
        self.executed_lines['filename'] = func_code.co_filename
        self.executed_lines['function_name'] = func_code.co_name

        try:
            func_lines, start_line = inspect.getsourcelines(unwrapped_func)
        except OSError as e:
            _get_logger().error(
                f"Cannot inspect source of '{func.__name__}': it appears to be wrapped by a decorator "
                f"that doesn't preserve source info. Try putting @annotate._method() as the innermost "
                f"decorator (closest to the function definition)."
            )
            raise OSError(
                f"Cannot get source lines for function '{func.__name__}'. "
                f"If this function is wrapped by another decorator, try reordering so @annotate._method() "
                f"is the innermost decorator."
            ) from e

        self.executed_lines['min_line'] = start_line
        self.executed_lines['max_line'] = start_line + len(func_lines) - 1

    def create_trace_function(self, skip_file, entry_hook=None):
        """Create a trace function for use with sys.settrace."""
        def trace_function(frame, event, arg):
            if event == 'call':
                code = frame.f_code
                if code.co_filename.split('/')[-1] == skip_file:
                    return trace_function

                if (code.co_filename == self.executed_lines['filename'] and
                        self.executed_lines['min_line'] <= frame.f_lineno <= self.executed_lines['max_line']):
                    namespace = frame_to_namespace(frame)
                    self.output_namespace = namespace

            return trace_function
        return trace_function

    # ── function input / output inspection ───────────────────────────────

    def inspect_function_inputs(self, func, args, kwargs):
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            self._check_for_active_traced_tensors(param_value, param_name)
            if param_name == 'self':
                continue
            param = sig.parameters[param_name]
            was_provided = (
                param_name in kwargs or
                param_names.index(param_name) < len(args)
            )
            if was_provided:
                self.add_input(param_name, param_name, param_value)
            else:
                self.default_kwargs[param_name] = param_value

    def inspect_function_outputs(self, func, result):
        if result is None:
            return
        return_names = extract_return_names(func)

        if isinstance(result, tuple) and len(return_names) == len(result):
            for i in range(len(result)):
                self.tag_data(result[i], return_names[i])
                if i < len(return_names):
                    output_name = return_names[i]
                else:
                    output_name = f"output{i+1}"
                self.add_output(output_name, output_name, result[i])
        else:
            if not len(return_names) == 1:
                raise Exception(
                    f"Error: {self.name} has {len(return_names)}"
                    " outputs, but only one output is detected")
            self.tag_data(result, return_names[0])
            self.add_output(return_names[0], return_names[0], result)

    # ── validation (re-entry) ────────────────────────────────────────────

    def validate_function_boundaries(self, func):
        """Validate that function boundaries match on re-entry."""
        unwrapped_func = inspect.unwrap(func)
        func_code = unwrapped_func.__code__
        func_lines, start_line = inspect.getsourcelines(unwrapped_func)

        new_boundaries = {
            'filename': func_code.co_filename,
            'function_name': func_code.co_name,
            'min_line': start_line,
            'max_line': start_line + len(func_lines) - 1
        }

        for key in ('filename', 'function_name', 'min_line', 'max_line'):
            if self.executed_lines.get(key) != new_boundaries[key]:
                original_data = {k: self.executed_lines.get(k) for k in (
                    'filename', 'function_name', 'min_line', 'max_line')}
                _get_logger().error(
                    f"Error: {self.name} seen twice but function boundaries do not match\n"
                    f"Original: {original_data}\n"
                    f"New: {new_boundaries}")
                raise Exception(
                    f"Error: {self.name} seen twice but function boundaries do not match\n"
                    f"Original: {original_data}\n"
                    f"New: {new_boundaries}")

        self.executed_lines['lines'] = set(
            range(start_line, start_line + len(func_lines)))

    def validate_function_inputs(self, func, args, kwargs):
        """Validate function inputs on re-entry against previously captured inputs."""
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            self._check_for_active_traced_tensors(param_value, param_name)
            if param_name == 'self':
                continue
            param = sig.parameters[param_name]
            was_provided = (
                param_name in kwargs or
                param_names.index(param_name) < len(args)
            )
            if was_provided:
                self.validate_input_and_update_tags(
                    param_name, param_name, param_value)

    def validate_function_outputs(self, func, result):
        """Validate function outputs on re-entry against previously captured outputs."""
        if result is None:
            return
        return_names = extract_return_names(func)

        if isinstance(result, tuple) and len(return_names) == len(result):
            for i in range(len(result)):
                self.tag_data(result[i], return_names[i])
                if i < len(return_names):
                    output_name = return_names[i]
                else:
                    output_name = f"output{i+1}"
                self.validate_output_and_update_tags(
                    output_name, output_name, result[i])
        else:
            if not len(return_names) == 1:
                raise Exception(
                    f"Error: {self.name} has {len(return_names)}"
                    " outputs, but only one output is detected")
            self.tag_data(result, return_names[0])
            self.validate_output_and_update_tags(
                return_names[0], return_names[0], result)

    # ── namespace capture / validation ───────────────────────────────────

    def capture_inputs_from_namespace(self, namespace):
        """Capture declared inputs from a namespace dictionary."""
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
        """Capture declared outputs from a namespace dictionary."""
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

    def validate_inputs_from_namespace(self, namespace):
        """Validate declared inputs against a namespace dictionary."""
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

    def validate_outputs_from_namespace(self, namespace):
        """Validate declared outputs against a namespace dictionary."""
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

    # ── buffer / constant snapshot ───────────────────────────────────────

    def snapshot_buffer_values(self, namespace):
        """Cache buffer and constant values from a namespace."""
        for buffer_name in self.register_buffers:
            value, _ = get_attribute_value_from_namespace(namespace, buffer_name)
            if buffer_name not in self.cached_buffer_values:
                self.cached_buffer_values[buffer_name] = safe_deepcopy(value)

        for constant_name in self.environment_constants:
            value, _ = get_attribute_value_from_namespace(namespace, constant_name)
            if constant_name not in self.cached_constant_values:
                self.cached_constant_values[constant_name] = safe_deepcopy(value)

    # ── internal helpers ─────────────────────────────────────────────────

    def _capture_specified_value_from_namespace(self, variable_name, namespace):
        """Extract a variable value from a namespace dictionary."""
        obj, final_variable_name = get_attribute_value_from_namespace(
            namespace, variable_name)
        self._check_for_active_traced_tensors(obj, variable_name)
        return final_variable_name, safe_deepcopy(obj)

    def _check_for_active_traced_tensors(self, data, variable_name, path=None):
        """Recursively check if data contains any active TracedTensor instances."""
        if not path:
            path = variable_name
        if isinstance(data, TracedTensor) and data.is_tracing:
            _get_logger().error(
                f"Cannot use TracedTensor as input to _method() '{self.name}'.\n"
                f"Variable '{variable_name}' (at {path}) contains an active TracedTensor "
                f"from node '{data.context}'.\n"
                f"\n"
                f"This happens when you try to use a TracedTensor created by input_tensors() "
                f"as input to code inside _method().\n"
                f"\n"
                f"You must call output_tensors() to finalize the TracedTensor node first"
            )
            raise Exception(
                f"Cannot use TracedTensor '{path}' as input to _method() '{self.name}'. "
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
