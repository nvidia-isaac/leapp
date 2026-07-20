#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import torch
import types
import linecache
import textwrap
import re
from torch.nn.modules.lazy import LazyModuleMixin
from leapp.utils.logging import _get_logger
from leapp.utils.utils import extract_source_from_line_range

from typing import Tuple, List, Dict


def get_module_template(name, parent_class, constant_attrs):
    if parent_class is None:
        bases = (torch.nn.Module,)
    elif isinstance(parent_class, type) and issubclass(parent_class, torch.nn.Module):
        bases = (parent_class,)
    else:
        bases = (torch.nn.Module, parent_class)

    def __init__(self, *args, **kwargs):
        # Call torch.nn.Module.__init__ directly to set up essential module internals
        # (_modules, _parameters, _buffers, etc.) without reinitializing the parent class.
        # This preserves any user-modified class variables. The actual attribute duplication
        # happens later in _duplicate_attributes().
        torch.nn.Module.__init__(self)
        self.saved_buffers: List[str] = []

    def add_buffer(self, name, value):
        if 'self.' in name:
            name = name.split('self.')[1]
        if name in self.saved_buffers:
            _get_logger().fatal(
                f"Buffer with name '{name}' was already registered",
                error_type=ValueError,
            )
        if name in self.__constant__:
            _get_logger().fatal(
                f"Buffer with name '{name}' was already registered as a constant",
                error_type=ValueError,
            )
        if hasattr(self, name):
            delattr(self, name)
        self.register_buffer(name, value)
        self.saved_buffers.append(name)

    def forward(self):
        raise NotImplementedError("The forward method is not implemented.")

    # Create the class dynamically using type()
    # Note: _core is intentionally not included here - it's dynamically bound
    # as an instance method in _compile_forward_function to avoid TorchScript
    # picking up a wrong signature from the class definition.
    ModuleTemplate = type(
        name,
        bases,
        {
            '__init__': __init__,
            'add_buffer': add_buffer,
            # '_core' : intentionally not included here - it's dynamically bound
            'forward': forward,
            '__constant__': list(),
        }
    )

    module_instance = ModuleTemplate()

    return module_instance

class ModuleBuilder:

    def __init__(self, node_context):
        self.node_context = node_context

    def __call__(self):
        # validation
        if self.node_context.input_namespace is None or self.node_context.output_namespace is None:
            _get_logger().fatal(
                f"Input or output namespace not found for {self.node_context.name}",
                error_type=Exception)

        # if the function or snippet is a class method, we need to create a module that inherits
        # from the class to preserve class functions
        input_namespace = self.node_context.input_namespace
        if 'self' in input_namespace:
            parent_class = type(input_namespace['self'])
        else:
            parent_class = None
        self.module_instance = get_module_template(
            self.node_context.name, parent_class, self.node_context.cached_constant_values)

        # setup the module with required attributes from the environment
        self._register_explicit_values()
        self._duplicate_attributes()

        # extract the body of the function
        body = self._extract_source_code()
        core_header, prepended_conditioning, return_statement, outputs_str = self._create_core_header_and_return()
        forward_header, forward_body, forward_return_statement = self._create_forward_header_and_body()
        
        #TODO: this can be removed once we are sure that the new method works with self. returns
        if self.node_context.from_function:
            # For functions, append outputs_str to all return statements
            modified_source, replacement_count = self._append_outputs_to_return_statements(
                body, outputs_str)
            body = modified_source
            if replacement_count != 0 or len(self.node_context._declared_outputs) == 0:
                return_statement = ""
            # if replacement count is 0, we use the artificial return statement created previously

        # stitch the _core function together:
        final_core_body = prepended_conditioning + body + return_statement
        indented_core_body = textwrap.indent(final_core_body, '    ')
        core_function_string = core_header + indented_core_body

        final_forward_body = forward_body + forward_return_statement
        indented_forward_body = textwrap.indent(final_forward_body, '    ')
        forward_function_string = forward_header + indented_forward_body

        # combine _core and forward functions
        function_string = core_function_string + "\n" + forward_function_string

        _get_logger().debug("\n"+function_string)

        self._compile_forward_function(function_string)

        return self.module_instance

    def _register_explicit_values(self):
        for constant_name, constant_value in self.node_context.cached_constant_values.items():
            if 'self.' in constant_name:
                constant_name = constant_name.split('self.')[1]
            setattr(self.module_instance, constant_name, constant_value)
            self.module_instance.__constant__.append(constant_name)
            _get_logger().debug(
                f"Set instance attribute (constant value): {constant_name} = {constant_value}")
        # Set default values as instance attributes (from stored dictionary)
        for default_name, default_value in self.node_context.default_kwargs.items():
            # Set as instance attribute - accessible as self.default_name
            setattr(self.module_instance, default_name, default_value)
            _get_logger().debug(
                f"Set instance attribute (default value): {default_name} = {default_value}")

        # add buffers to the module
        for buffer_name in self.node_context.register_buffers:
            value = self.node_context.cached_buffer_values[buffer_name]
            self.module_instance.add_buffer(buffer_name, value)

        if len(self.module_instance.saved_buffers) > 0:
            _get_logger().info("Created the following buffers:")
            for buffer_name in self.module_instance.saved_buffers:
                _get_logger().info(
                    f"  - self.{buffer_name}: initialized as {getattr(self.module_instance, buffer_name)}")

    def _duplicate_attributes(self):
        # copy all the attributes from the original object to the module
        input_namespace = self.node_context.input_namespace
        if 'self' in input_namespace:
            original_obj = input_namespace['self']

            # Warning #1: Lazy modules may not transfer correctly
            if isinstance(original_obj, LazyModuleMixin) and original_obj.has_uninitialized_params():
                _get_logger().error(
                    "Parent class contains uninitialized lazy parameters. "
                    "Export may fail or produce incorrect results.")

            # Warning #3: Check for registered hooks that may not transfer correctly
            hook_attrs = ['_forward_hooks', '_backward_hooks', '_forward_pre_hooks', '_backward_pre_hooks']
            for hook_attr in hook_attrs:
                if hasattr(original_obj, hook_attr) and len(getattr(original_obj, hook_attr)) > 0:
                    _get_logger().error(
                        f"Parent class has registered {hook_attr}. These may not transfer correctly.")

            for attr_name in dir(original_obj):
                # we don't copy any private attributes or register buffers or registered constants
                if attr_name.startswith('__') or attr_name.endswith('__') or \
                        f"self.{attr_name}" in self.node_context.register_buffers or \
                        f"{attr_name}" in self.module_instance.__constant__:
                    continue

                try:
                    value = getattr(original_obj, attr_name)
                    _get_logger().debug(f"Setting attribute: {attr_name}")

                    # Check if it's a property (read-only or not)
                    if hasattr(type(original_obj), attr_name):
                        prop = getattr(type(original_obj), attr_name)
                        if isinstance(prop, property):
                            # Bypass property setter by setting directly in __dict__
                            self.module_instance.__dict__[attr_name] = value

                    elif isinstance(value, torch.nn.Module):
                        self.module_instance.add_module(attr_name, value)

                    elif not callable(value):
                        self.module_instance.__dict__[attr_name] = value
                    elif callable(value):
                        # Check if value is already a bound method
                        if isinstance(value, types.MethodType):
                            # Use the underlying function, not the bound method
                            func = value.__func__
                            bound_method = types.MethodType(func, self.module_instance)
                        else:
                            bound_method = types.MethodType(value, self.module_instance)
                        setattr(self.module_instance, attr_name, bound_method)
                except Exception as e:
                    _get_logger().error(
                        f"Error setting attribute {attr_name}: {e}")

            # Fix #2: Also copy slotted attributes (not in __dict__)
            for cls in type(original_obj).__mro__:
                if hasattr(cls, '__slots__'):
                    for slot in cls.__slots__:
                        if slot.startswith('__') or slot.endswith('__'):
                            continue
                        if hasattr(original_obj, slot) and not hasattr(self.module_instance, slot):
                            try:
                                value = getattr(original_obj, slot)
                                setattr(self.module_instance, slot, value)
                                _get_logger().debug(f"Copied slotted attribute: {slot}")
                            except Exception as e:
                                _get_logger().error(f"Error copying slotted attribute {slot}: {e}")

    def _extract_source_code(self):
        """Extract the source code body for the traced function or block.
        
        Uses the unified extract_source_from_line_range function which handles
        both function decorators and block contexts, and properly captures
        multiline statements.
        """
        executed_lines = self.node_context.executed_lines
        
        source_code, message = extract_source_from_line_range(
            executed_lines,
            self.node_context.name,
            is_function=self.node_context.from_function
        )
        
        if source_code:
            _get_logger().info(message)
        else:
            _get_logger().fatal(
                message,
                error_type=Exception)

        return source_code

    def _create_core_header_and_return(self):
        """Create header, body prepend (data patching), and return statement for _core function."""
        # Output type annotation for _core
        output_types = [
            parameter_description.dtype for parameter_description in self.node_context.output_formats]
        if len(output_types) == 0:
            output_types_str = "None"
        elif len(output_types) == 1:
            output_types_str = output_types[0]
        else:
            output_types_str = f"Tuple[{', '.join(output_types)}]"
        
        # Core function needs type annotations for TorchScript
        core_inputs = ', '.join([f"{input_val.name_str}: {input_val.dtype}" 
                                  for input_val in self.node_context.input_formats])
        header = f"def _core(self, {core_inputs}) -> {output_types_str}:\n"

        # Build data patching string (prepended to core body)
        prepended_conditioning = ""
        
        # Handle input name transformations (e.g., "name_str" -> "name_raw")
        for input in self.node_context.input_formats:
            if input.name_raw != input.name_str:
                prepended_conditioning += f"{input.name_raw} = {input.name_str}\n"

        # Handle constants, buffers, and default kwargs
        targets = self.node_context.environment_constants | self.node_context.register_buffers | set(
            self.node_context.default_kwargs.keys())
        for const_name in targets:
            if "self." in const_name or const_name == "self":
                continue  # special case for any class variables, no rename required
            prepended_conditioning += f"{const_name} = self.{const_name}\n"

        # only include outputs that are declared by user.
        outputs = ", ".join([output.name_raw
                            for output in self.node_context.output_formats
                            if output.name_raw in self.node_context._declared_outputs])
        return_statement = f"\nreturn {outputs}\n" if outputs else ""

        return header, prepended_conditioning, return_statement, outputs

    def _create_forward_header_and_body(self):
        """Create header and body for forward function."""
        # Forward function accepts flat tensor inputs (no type annotations)
        forward_input_names = ", ".join(
            [tensor_desc.name for tensor_desc in self.node_context.inputs])
        
        header = f"def forward(self, {forward_input_names}):\n"

        prepended_conditioning = ""
        # Handle input packing: convert flat tensor inputs to expected nested structures
        for param_format in self.node_context.input_formats:
            packing_expr = param_format.packed_tensor_expr
            if packing_expr:
                prepended_conditioning += packing_expr + "\n"
        
        appended_conditioning = ""
        for param_format in self.node_context.output_formats:
            appended_conditioning += param_format.unpacked_tensor_expr + "\n"

        # Build forward function body with explicit unpacking
        core_output_names = [output.name_str for output in self.node_context.output_formats]
        core_input_names = [input.name_str for input in self.node_context.input_formats]
        core_input_names_str = ', '.join(core_input_names)
        forward_output_names = [output.name for output in self.node_context.outputs]

        if len(core_output_names) == 0:
            body = f"self._core({core_input_names_str})\n"
        elif len(core_output_names) == 1:
            body = f"{core_output_names[0]} = self._core({core_input_names_str})\n"
        else:
            output_names_str = ", ".join(core_output_names)
            body = f"({output_names_str}) = self._core({core_input_names_str})\n"

        body = prepended_conditioning + body + appended_conditioning
        
        # Build return statement based on the actual number of forward outputs (after unpacking)
        if len(forward_output_names) == 0:
            return_statement = ""
        elif len(forward_output_names) == 1:
            return_statement = f"return {forward_output_names[0]}\n"
        else:
            forward_output_names_str = ", ".join(forward_output_names)
            return_statement = f"return {forward_output_names_str}\n"
        
        return header, body, return_statement

    def _append_outputs_to_return_statements(self, source_code, outputs_str):
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

    def _compile_forward_function(self, function_string):
        filename = f'generated_{self.node_context.name}_{id(self.module_instance)}_function'
        try:
            code = compile(function_string, filename, "exec")

            # recreate the environment of the function using stored namespace
            namespace = {
                'Tensor': torch.Tensor,
                "Tuple": Tuple,
                "List": List,
                "Dict": Dict,
                "NoneType": type(None),
            }
            if self.node_context.input_namespace is not None:
                namespace.update(self.node_context.input_namespace)

            exec(code, namespace)
        except Exception as e:
            _get_logger().fatal(
                f"Error compiling function: {e}\n{function_string}",
                error_type=type(e),
                cause=e)

        _core = namespace['_core']
        forward = namespace['forward']

        lines = [line + '\n' for line in function_string.splitlines()]
        linecache.cache[filename] = (
            len(function_string), None, lines, filename)
        
        # Set methods on the class (not instance) so TorchScript can recognize them
        module_class = type(self.module_instance)
        module_class._core = _core
        module_class.forward = forward
