import torch
import types
import linecache
import textwrap
import re
import ast
from leapp._logging import _get_logger

from typing import Tuple, List, Dict


def get_module_template(name, parent_class, constant_attrs):
    if parent_class is None:
        bases = (torch.nn.Module,)
    elif isinstance(parent_class, type) and issubclass(parent_class, torch.nn.Module):
        bases = (parent_class,)
    else:
        bases = (torch.nn.Module, parent_class)

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.saved_buffers: List[str] = []

    def add_buffer(self, name, value):
        if 'self.' in name:
            name = name.split('self.')[1]
        if name in self.saved_buffers:
            raise ValueError(
                f"Buffer with name '{name}' was already registered")
        if name in self.__constant__:
            raise ValueError(
                f"Buffer with name '{name}' was already registered as a constant")
        if hasattr(self, name):
            delattr(self, name)
        self.register_buffer(name, value)
        self.saved_buffers.append(name)

    def forward(self):
        raise NotImplementedError("The forward method is not implemented.")

    # Create the class dynamically using type()
    ModuleTemplate = type(
        name,
        bases,
        {
            '__init__': __init__,
            'add_buffer': add_buffer,
            'forward': forward,
            '__constant__': list(),
        }
    )

    module_instance = ModuleTemplate()

    return module_instance

# def build_input_from_raw_tensors(raw_tensors, input_formats):
#     def builder_function(self, )


class ModuleBuilder:

    def __init__(self, node_context):
        self.node_context = node_context

    def __call__(self):
        # validation
        if self.node_context.input_frame is None or self.node_context.output_frame is None:
            raise Exception(
                f"Input or output frame not found for {self.node_context.name}")

        # if the function or snippet is a class method, we need to create a module that inherits
        # from the class to preserve class functions
        if 'self' in self.node_context.input_frame.f_locals:
            parent_class = type(self.node_context.input_frame.f_locals['self'])
        else:
            parent_class = None
        self.module_instance = get_module_template(
            self.node_context.name, parent_class, self.node_context.cached_constant_values)

        # setup the module with required attributes from the environment
        self._register_explicit_values()
        self._duplicate_attributes()

        # extract the body of the function
        body = self._extract_source_code()
        header, return_statement, outputs_str = self._create_header_and_return_statement()
        if self.node_context.from_function:
            # For functions, append outputs_str to all return statements
            modified_source, replacement_count = self._append_outputs_to_return_statements(
                body, outputs_str)
            body = modified_source
            if replacement_count != 0 or len(self.node_context._declared_outputs) == 0:
                return_statement = ""
            # if replacement count is 0, we use the artificial return statement created previously

        # modifications to the body of the function
        prepended_conditioning = self._create_data_patching_string()

        # stitch the function together:
        function_body = prepended_conditioning + body + return_statement

        # Add 4-space indentation to all lines of function body
        indented_function_body = textwrap.indent(function_body, '    ')

        # add the headder
        function_string = header + indented_function_body

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
                    f"  - self.{buffer_name}: intialized as {getattr(self.module_instance, buffer_name)}")

    def _duplicate_attributes(self):
        # copy all the attributes from the original object to the module
        if 'self' in self.node_context.input_frame.f_locals:
            original_obj = self.node_context.input_frame.f_locals['self']
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
                        bound_method = types.MethodType(
                            value, self.module_instance)
                        setattr(self.module_instance, attr_name, bound_method)
                except Exception as e:
                    _get_logger().error(
                        f"Error setting attribute {attr_name}: {e}")

    def _extract_source_code(self):
        executed_lines = self.node_context.executed_lines

        if executed_lines['filename'] is None:
            return ""

        try:
            filename = executed_lines['filename']
            min_line = executed_lines['min_line']
            max_line = executed_lines['max_line']

            # Read the source file
            with open(filename, 'r') as f:
                source_lines = f.readlines()

            # Extract lines from min_line to max_line (convert to 0-based indexing)
            start_idx = min_line - 1
            end_idx = max_line  # max_line is inclusive, so we don't subtract 1

            if start_idx < 0 or end_idx > len(source_lines):
                print(
                    f"\033[93mWarning {self.node_context.name}: Line range {min_line}-{max_line}"
                    f" is out of bounds for file {filename}\033[0m")
                return ""

            # Determine the start of the executable body using AST (robust for multiline headers)
            file_source = ''.join(source_lines)
            body_start_idx = None

            try:
                tree = ast.parse(file_source)

                best_candidate = None
                best_start = -1

                if self.node_context.from_function:
                    wanted = (ast.FunctionDef, ast.AsyncFunctionDef)
                else:
                    wanted = (ast.With, ast.AsyncWith)

                # Find the function that contains our line range
                for n in ast.walk(tree):
                    if isinstance(n, wanted):
                        node_start = getattr(n, 'lineno', None)
                        node_end = getattr(n, 'end_lineno', None)
                        if node_start is None:
                            continue
                        if node_end is None:
                            if getattr(n, 'body', None):
                                last_child = n.body[-1]
                                node_end = getattr(last_child, 'end_lineno', getattr(
                                    last_child, 'lineno', node_start))
                            else:
                                node_end = node_start

                        # Check if this function overlaps with our line range
                        # (handles decorators by checking if function def line or any body line is in range)
                        if (min_line <= node_start <= max_line) or (node_start <= min_line <= node_end):
                            if node_start > best_start:
                                best_candidate = n
                                best_start = node_start

                if best_candidate is not None and getattr(best_candidate, 'body', None):
                    body_start_line = getattr(
                        best_candidate.body[0], 'lineno', None)
                    if body_start_line is not None:
                        body_start_idx = body_start_line - 1
            except Exception:
                body_start_idx = None

            if body_start_idx is None:
                # Fallback: scan to the end of the logical header by balancing symbols and strings
                open_paren = open_brack = open_brace = 0
                in_string = False
                string_char = ''
                header_end_line_idx = start_idx
                i = start_idx
                while i < end_idx:
                    line = source_lines[i]
                    j = 0
                    while j < len(line):
                        ch = line[j]
                        if not in_string and ch in ('"', "'"):
                            in_string = True
                            string_char = ch
                        elif in_string and ch == string_char and (j == 0 or line[j-1] != '\\'):
                            in_string = False
                            string_char = ''
                        elif not in_string:
                            if ch == '(':
                                open_paren += 1
                            elif ch == ')':
                                open_paren = max(0, open_paren-1)
                            elif ch == '[':
                                open_brack += 1
                            elif ch == ']':
                                open_brack = max(0, open_brack-1)
                            elif ch == '{':
                                open_brace += 1
                            elif ch == '}':
                                open_brace = max(0, open_brace-1)
                            elif ch == ':' and open_paren == 0 and open_brack == 0 and open_brace == 0:
                                header_end_line_idx = i
                                i = i + 1
                                break
                        j += 1
                    else:
                        i += 1
                        continue
                    break

                body_start_idx = max(header_end_line_idx + 1, start_idx)

            start_idx = min(max(body_start_idx, 0), len(source_lines))

            # Extract the relevant lines
            extracted_lines = source_lines[start_idx:end_idx]

            if extracted_lines:
                dedented_code = textwrap.dedent(''.join(extracted_lines))
                source_code = dedented_code.rstrip()
            else:
                source_code = ""
            _get_logger().info(
                f"Extracted code from file: {filename}, function: {executed_lines['function_name']}")

        except Exception as e:
            _get_logger().error(
                f"Error extracting source from line range: {e}")
            source_code = ""

        if source_code == "":
            raise Exception(
                f"No source code found for {self.node_context.name}")

        return source_code

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

    def _compile_forward_function(self, function_string):
        filename = f'generated_{self.node_context.name}_function'
        try:
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
            _get_logger().error(f"Error compiling function: {e}")
            _get_logger().error(function_string)
            raise e

        forward = namespace['forward']

        lines = [line + '\n' for line in function_string.splitlines()]
        linecache.cache[filename] = (
            len(function_string), None, lines, filename)
        self.module_instance.forward = types.MethodType(
            forward, self.module_instance)


if __name__ == "__main__":
    module_builder = ModuleBuilder("TestModule", None, {})
    print(module_builder.module)
