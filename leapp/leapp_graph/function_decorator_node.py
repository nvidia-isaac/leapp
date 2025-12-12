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
)
from leapp._logging import _get_logger
from leapp.leapp_graph.block_context_node import BlockContextNode
import inspect


class FunctionDecoratorNode(BlockContextNode):
    def __init__(self, name, node_index, backend=None, use_trace=False,
                 backend_params=None, inputs=None, outputs=None,
                 environment_constants=None, register_buffers=None):
        super().__init__(name, node_index, backend, use_trace, backend_params, inputs, outputs, environment_constants, register_buffers)
        # node settings
        self.from_function = True
        
    def compile_model(self):
        try:
            self.compiled_model = self.export_backend.compile()
        except Exception as e:
            _get_logger().error(f"Error compiling model: {e}")
            raise e
    
    def compile_trace(self, func, *args):
        func_code = func.__code__
        self.executed_lines['filename'] = func_code.co_filename
        self.executed_lines['function_name'] = func_code.co_name

        # Get the line range of the function
        func_lines, start_line = inspect.getsourcelines(func)
        self.executed_lines['min_line'] = start_line
        self.executed_lines['max_line'] = start_line + \
            len(func_lines) - 1

        # Initialize the lines set with all function lines
        self.executed_lines['lines'] = set(
            range(start_line, start_line + len(func_lines)))

    def create_trace_function(self, skip_file):
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

                # Save frame if function is within the traced function's line range
                if (code.co_filename == self.executed_lines['filename'] and
                        self.executed_lines['min_line'] <= frame.f_lineno <= self.executed_lines['max_line']):
                    if self.input_frame is None:
                        self.input_frame = frame  # we will only store input frame once
                        # store buffer values upon entering the function
                        self.snapshot_buffer_values(frame)
                    # Keep on updating output frame
                    self.output_frame = frame

            return trace_function
        return trace_function

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

        # extract custom returns from the environment variables
