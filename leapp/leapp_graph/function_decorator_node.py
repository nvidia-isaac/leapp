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
    frame_to_namespace,
)
from leapp._logging import _get_logger
from leapp.leapp_graph.block_context_node import BlockContextNode
import inspect


class FunctionDecoratorNode(BlockContextNode):
    def __init__(self, name, node_index, backend=None,
                 backend_params=None, inputs=None, outputs=None,
                 environment_constants=None, register_buffers=None):
        super().__init__(name, node_index, backend, backend_params,
                         inputs, outputs, environment_constants, register_buffers)
        # node settings
        self.from_function = True

    def compile_trace(self, func, *args):
        # Unwrap decorated functions to get to the original source
        unwrapped_func = inspect.unwrap(func)

        # Warn if the function is wrapped - tracing won't trace through the wrapper
        if unwrapped_func is not func:
            _get_logger().warning(
                f"Function '{func.__name__}' is wrapped by another decorator (e.g., @torch.no_grad()). "
                f"Tracing will only capture the inner function's code, not the wrapper's behavior."
            )

        func_code = unwrapped_func.__code__
        self.executed_lines['filename'] = func_code.co_filename
        self.executed_lines['function_name'] = func_code.co_name

        # Get the line range of the function
        try:
            func_lines, start_line = inspect.getsourcelines(unwrapped_func)
        except OSError as e:
            # Source not available (e.g., C extension, built-in, or dynamically generated)
            _get_logger().error(
                f"Cannot inspect source of '{func.__name__}': it appears to be wrapped by a decorator "
                f"that doesn't preserve source info. Try putting @annotate.method() as the innermost "
                f"decorator (closest to the function definition)."
            )
            raise OSError(
                f"Cannot get source lines for function '{func.__name__}'. "
                f"If this function is wrapped by another decorator, try reordering so @annotate.method() "
                f"is the innermost decorator."
            ) from e

        self.executed_lines['min_line'] = start_line
        self.executed_lines['max_line'] = start_line + len(func_lines) - 1

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
                    f"Error: {self.name} seen twice but block boundaries do not match\n"
                    f"Original: {original_data}\n"
                    f"New: {new_boundaries}")
                raise Exception(
                    f"Error: {self.name} seen twice but block boundaries do not match\n"
                    f"Original: {original_data}\n"
                    f"New: {new_boundaries}")

        # Initialize the lines set with all function lines
        self.executed_lines['lines'] = set(
            range(start_line, start_line + len(func_lines)))

    def create_trace_function(self, skip_file, entry_hook=None):
        """Create and return the trace function for function decorator tracing.

        Args:
            skip_file: Filename to skip when tracing (e.g., export_manager.py)

        Returns:
            A trace function suitable for use with sys.settrace
        """
        def trace_function(frame, event, arg):
            if event == 'call':
                code = frame.f_code
                # Skip tracing the specified file
                if code.co_filename.split('/')[-1] == skip_file:
                    return trace_function

                # Save namespace if function is within the traced function's line range
                # we need to repeatedly save the message namespace. This is to make sure we can capture the last outputs of the function before it returns
                if (code.co_filename == self.executed_lines['filename'] and
                        self.executed_lines['min_line'] <= frame.f_lineno <= self.executed_lines['max_line']):
                    # Convert frame to namespace for flexibility
                    namespace = frame_to_namespace(frame)
                    # Keep on updating output namespace
                    self.output_namespace = namespace

            return trace_function
        return trace_function

    def inspect_function_inputs(self, func, args, kwargs):
        # NOTE: Don't use inspect.unwrap() here!
        # inspect.signature() already follows __wrapped__ AND handles bound methods correctly.
        # Using unwrap() would lose the bound method's self binding.
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        # bind
        bound_args = sig.bind(*args, **kwargs)
        # apply default values
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            self._check_for_active_traced_tensors(param_value, param_name)
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
                    " outputs, but only one output is detectd")
            self.tag_data(result, return_names[0])
            self.add_output(return_names[0], return_names[0], result)

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

        # extract custom returns from the environment variables
