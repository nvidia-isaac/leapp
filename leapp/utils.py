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
import inspect
import ast
import re
import textwrap
import copy
import collections.abc
from dataclasses import dataclass, field
from typing import Optional, Any, Dict


def extract_source_from_line_range(executed_lines, from_function, context_name):
    """
    Extract source code from a traced line range.

    Args:
        executed_lines: Dictionary with keys 'filename', 'min_line', 'max_line', 'function_name'
        from_function: Boolean indicating if the source is from a function
        context_name: Name of the context/node for logging purposes

    Returns:
        str: Extracted source code or empty string if extraction fails
    """
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
                f"\033[93mWarning {context_name}: Line range {min_line}-{max_line}"
                f" is out of bounds for file {filename}\033[0m")
            return ""

        # Determine the start of the executable body using AST (robust for multiline headers)
        file_source = ''.join(source_lines)
        body_start_idx = None

        try:
            tree = ast.parse(file_source)
            target_line = min_line

            def within(node):
                node_start = getattr(node, 'lineno', None)
                node_end = getattr(node, 'end_lineno', None)
                if node_start is None:
                    return False
                if node_end is None:
                    if getattr(node, 'body', None):
                        last_child = node.body[-1]
                        node_end = getattr(last_child, 'end_lineno', getattr(
                            last_child, 'lineno', node_start))
                    else:
                        node_end = node_start
                return node_start <= target_line <= node_end

            best_candidate = None
            best_start = -1

            if from_function:
                wanted = (ast.FunctionDef, ast.AsyncFunctionDef)
            else:
                wanted = (ast.With, ast.AsyncWith)

            for n in ast.walk(tree):
                if isinstance(n, wanted) and within(n):
                    cand_start = getattr(n, 'lineno', 0)
                    if cand_start >= best_start and cand_start <= target_line:
                        best_candidate = n
                        best_start = cand_start

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
        print(
            f"Extracted code from file: {filename}, function: {executed_lines['function_name']}")

    except Exception as e:
        print(f"Error extracting source from line range: {e}")
        source_code = ""

    return source_code


class CompactYamlList(list):
    """Custom list class for tensor shapes that enables compact YAML formatting
    This class will behave exactly like a list. The only difference is that it will be
    formatted in a more compact way in the yaml file.

    """
    pass


@dataclass
class TensorDescription:
    """Dataclass for describing tensor inputs/outputs in the computational graph."""
    name: str
    dtype: str
    shape: CompactYamlList
    type: str = "tensor"
    tag: Optional[str] = None
    value: Optional[Any] = None  # Will store torch.Tensor

    def dict(self, ignore_tag=True, ignore_value=True) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        result = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
            "type": self.type
        }
        if self.tag is not None and not ignore_tag:
            result["tag"] = self.tag
        if self.value is not None and not ignore_value:
            result["value"] = self.value
        return result

    def change_name(self, new_name: str):
        """
        Change the name of the tensor description.
        """
        self.name = new_name

    @property
    def name_str(self) -> str:
        """Return just the name as a string."""
        return self.name


def verify_data_exact_match(source_data, target_data):
    if type(source_data) is not type(target_data):
        return False

    if isinstance(source_data, torch.Tensor):
        if source_data.shape != target_data.shape:
            return False
        if source_data.dtype != target_data.dtype:
            return False
        if source_data.device != target_data.device:
            return False
        if not torch.equal(source_data, target_data):
            return False

    elif isinstance(source_data, list):
        if len(source_data) != len(target_data):
            return False
        for source_item, target_item in zip(source_data, target_data):
            if not verify_data_exact_match(source_item, target_item):
                return False
    elif isinstance(source_data, dict):
        if source_data.keys() != target_data.keys():
            return False
        for key, source_item in source_data.items():
            if not verify_data_exact_match(source_item, target_data[key]):
                return False
    else:
        if source_data != target_data:
            return False

    return True


def extract_return_names(func):
    """Extract variable names from return statements in the function."""
    try:
        source = inspect.getsource(func)
        # Remove leading indentation to make it valid Python code
        dedented_source = textwrap.dedent(source)
        tree = ast.parse(dedented_source)

        # Collect all return statements with their names
        return_statements = []

        # Find all return statements
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.value:
                statement_names = []

                if isinstance(node.value, ast.Tuple):
                    # Multiple return values: return var1, var2
                    for i, elt in enumerate(node.value.elts):
                        if isinstance(elt, ast.Name):
                            statement_names.append(elt.id)
                        else:
                            # Complex expression, use generic name
                            statement_names.append(f"output{i+1}")
                elif isinstance(node.value, ast.Name):
                    # Single return value: return var1
                    statement_names.append(node.value.id)
                else:
                    # Complex expression, use generic name
                    statement_names.append("output1")

                return_statements.append(statement_names)

        if not return_statements:
            return ["output1"]

        # Check consistency in number of return values
        return_counts = [len(stmt) for stmt in return_statements]
        unique_counts = set(return_counts)

        if len(unique_counts) > 1:
            # Different number of return values - this is an error
            count_details = {count: return_counts.count(
                count) for count in unique_counts}
            raise Exception(
                f"Function {func.__name__} has inconsistent return statements with different numbers of values: "
                f"{count_details}. All return statements must return the same number of values."
            )

        # All return statements have the same number of values
        num_returns = return_counts[0]

        if len(return_statements) > 1:
            # Multiple return statements - log warning and intelligently merge
            print(f"Warning: Function {func.__name__} has {len(return_statements)} return statements. "
                  f"This may cause unexpected behavior if the return statements return different "
                  f"data types, shapes, or dtypes.")

        # Intelligently merge return names by position, prioritizing named over generic
        final_names = []
        for pos in range(num_returns):
            # Collect names at this position from all return statements
            names_at_position = [stmt[pos] for stmt in return_statements]

            # Separate named variables from generic ones
            named_vars = [
                name for name in names_at_position if not name.startswith("output")]
            generic_vars = [
                name for name in names_at_position if name.startswith("output")]

            # Choose the best name for this position
            if named_vars:
                # Prefer named variables, take the first unique one
                chosen_name = named_vars[0]
            elif generic_vars:
                # Only generic names available
                chosen_name = generic_vars[0]
            else:
                # Fallback
                chosen_name = f"output{pos+1}"

            final_names.append(chosen_name)

        return final_names

    except Exception as e:
        # Fallback to generic names if parsing fails
        print(
            f"Error extracting return names from {func.__name__}, reverting to defaults: {e}")
        return ["output1"]


def map_from_torch_dtype(notation):
    if notation == "prefix":
        return {
            torch.float64: "kFloat64",
            torch.float32: "kFloat32",
            torch.float16: "kFloat16",
            torch.int32: "kInt32",
            torch.int64: "kInt64",
            torch.uint8: "kUInt8",
            torch.int8: "kInt8",
            torch.bool: "kBool",
        }
    elif notation == "python":
        return {
            torch.float64: "float64",
            torch.float32: "float32",
            torch.float16: "float16",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.uint8: "uint8",
            torch.int8: "int8",
            torch.bool: "bool",
        }
    else:
        raise ValueError(f"Unsupported notation: {notation}")


def describe_io_helper(data, name_str, dtype_notation):
    data_description = []
    if isinstance(data, list):
        io_format = []
        for idx, item in enumerate(data):
            child_description, child_format = describe_io_helper(
                item, name_str+str(idx), dtype_notation)
            io_format.append(child_format)
            data_description.extend(child_description)
        return data_description, io_format
    elif isinstance(data, dict):
        io_format = {}
        for k, v in data.items():
            child_description, child_format = describe_io_helper(
                v, name_str+k, dtype_notation)

            io_format[k] = child_format
            data_description.extend(child_description)
    elif isinstance(data, torch.Tensor):
        tag = None
        if hasattr(data, 'export_manager_tag'):
            tag = data.export_manager_tag

        # Create TensorDescription dataclass instance
        tensor_desc = TensorDescription(
            name=name_str,
            dtype=map_from_torch_dtype(dtype_notation)[data.dtype],
            shape=CompactYamlList(data.shape),
            type="tensor",
            tag=tag,
            value=safe_deepcopy(data)
        )

        # Return as a list containing the dataclass (for now, keep compatibility)
        data_description = [tensor_desc]
        io_format = tensor_desc  # Plain string
    else:
        # For non-tensor data types
        data_desc = TensorDescription(
            name=name_str,
            dtype=None,
            shape=CompactYamlList([]),
            type=type(data),
            tag=None,
            value=safe_deepcopy(data)
        )
        data_description = [data_desc]
        io_format = data_desc

    return data_description, io_format


def describe_io(name, data, dtype_notation="python"):
    data_description, io_format = describe_io_helper(
        data, name, dtype_notation)
    return data_description, io_format


def _resolve_tensor_descriptions(io_format, extractor):
    """
    Recursively resolve TensorDescription objects using a custom extractor function.

    This function traverses through nested structures (lists, dicts, etc.) and
    applies the extractor function to any TensorDescription objects found.

    Args:
        io_format: The object to recursively process (can be TensorDescription, list, dict, or any other type)
        extractor: A function that takes a TensorDescription and returns the desired value

    Returns:
        The processed object with TensorDescription objects replaced by extracted values
    """
    if isinstance(io_format, TensorDescription):
        return extractor(io_format)
    elif isinstance(io_format, list):
        return [_resolve_tensor_descriptions(item, extractor) for item in io_format]
    elif isinstance(io_format, dict):
        return {key: _resolve_tensor_descriptions(value, extractor) for key, value in io_format.items()}
    else:
        return io_format


def resolve_tensor_descriptions_to_names(io_format):
    """
    Recursively resolve TensorDescription objects to their string names.

    Args:
        io_format: The object to recursively process

    Returns:
        The processed object with TensorDescription objects replaced by their string names
    """
    return _resolve_tensor_descriptions(io_format, lambda td: td.name_str)


def resolve_tensor_descriptions_to_values(io_format):
    """
    Recursively resolve TensorDescription objects to their values.

    Args:
        io_format: The object to recursively process

    Returns:
        The processed object with TensorDescription objects replaced by their values
    """
    return _resolve_tensor_descriptions(io_format, lambda td: td.value)


def safe_deepcopy(data):
    if isinstance(data, torch.Tensor):
        return data.clone()

    elif isinstance(data, list):
        return [safe_deepcopy(item) for item in data]
    elif isinstance(data, dict):
        return {k: safe_deepcopy(v) for k, v in data.items()}
    else:
        return copy.deepcopy(data)


def build_function_header(text, name):
    """
    Extract function header from text and replace function name with the provided name.
    Also inject 'self' as first parameter if not present.

    Args:
        text (str): Text containing one Python function
        name (str): The name to use for the function

    Returns:
        str: Single-line function header with specified name and self as first parameter, or None if not found
    """
    # Find the function definition
    match = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', text)
    if not match:
        return None

    # Find the function header by balancing parentheses
    start = match.start()
    header_end = find_header_end(text, start)
    if header_end is None:
        return None

    # Extract and clean the header
    raw_header = text[start:header_end + 1]
    clean_header = clean_header_to_single_line(raw_header)

    # Replace the function name with the provided name
    clean_header = re.sub(
        r'def\s+[a-zA-Z_]\w*\s*\(', f'def {name}(', clean_header)

    # Add self if not already present
    return add_self_if_needed(clean_header) + '\n'


def find_header_end(text, start_pos):
    """Find the end of function header (the colon after balanced parentheses)"""
    paren_count = 0
    in_string = False
    string_char = None
    found_params = False

    for i in range(start_pos, len(text)):
        char = text[i]

        # Handle string literals (simple approach)
        if not in_string and char in ('"', "'"):
            in_string = True
            string_char = char
        elif in_string and char == string_char and text[i-1:i] != '\\':
            in_string = False
            string_char = None
        elif in_string:
            continue

        # Handle parentheses
        if char == '(':
            paren_count += 1
            found_params = True
        elif char == ')':
            paren_count -= 1
        elif char == ':' and paren_count == 0 and found_params:
            return i

    return None


def clean_header_to_single_line(header):
    """Convert multi-line header to clean single line"""
    # Remove extra whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', header.strip())

    # Fix spacing around special characters
    cleaned = re.sub(r'\s*\(\s*', '(', cleaned)
    cleaned = re.sub(r'\s*\)\s*', ')', cleaned)
    cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
    cleaned = re.sub(r'\s*->\s*', ' -> ', cleaned)
    cleaned = re.sub(r'\s*:\s*$', ':', cleaned)

    return cleaned


def add_self_if_needed(header):
    """Add self parameter if not already present"""
    # Extract parameters between parentheses
    match = re.match(r'(.*?\()([^)]*)(\).*)', header)
    if not match:
        return header

    before, params, after = match.groups()
    params = params.strip()

    # Check if self is already the first parameter
    if not params:
        # No parameters - add self
        return f"{before}self{after}"

    # Get first parameter name (remove type hints and defaults)
    first_param = params.split(',')[0].strip()
    param_name = first_param.split(':')[0].split('=')[0].strip().lstrip('*')

    if param_name == 'self':
        # Self already present
        return header
    else:
        # Add self as first parameter
        return f"{before}self, {params}{after}"


def get_atrribute_value_from_frame(frame, attr_name):
    if "." in attr_name:
        parts = attr_name.split(".")
        final_attr_name = parts[-1]
        if parts[0] in frame.f_locals:
            obj = frame.f_locals[parts[0]]
        elif parts[0] in frame.f_globals:
            obj = frame.f_globals[parts[0]]
        else:
            raise Exception(
                f"Variable '{parts[0]}' not found in frame locals or globals")

        for attr in parts[1:]:
            try:
                obj = getattr(obj, attr)
            except Exception as e:
                raise Exception(
                    f"Error attempting to find {attr_name}, "
                    f"failed to get attribute {attr}\n",
                    e)

    else:
        if attr_name in frame.f_locals:
            obj = frame.f_locals[attr_name]
        elif attr_name in frame.f_globals:
            obj = frame.f_globals[attr_name]
        else:
            raise Exception(
                f"Variable '{attr_name}' not found in frame locals or globals")
        final_attr_name = attr_name

    return obj, final_attr_name


#########################################################
# Class Templates
#########################################################


def create_module(name, parent_class):
    if parent_class is None:
        bases = (torch.nn.Module,)
    else:
        bases = (torch.nn.Module, parent_class)

    def __init__(self, *args, **kwargs):
        # Initialize all parent classes
        super(ModuleTemplate, self).__init__(*args, **kwargs)
        self.saved_buffers = []

    def add_buffer(self, name, value):
        # verify no duplicate buffer names
        if name in self.saved_buffers:
            raise ValueError(f"Buffer {name} already registered")
        # clean up the input name
        if 'self.' in name:
            name = name.split('self.')[1]
        # clean up preexisting attributes
        if hasattr(self, name):
            delattr(self, name)
        # register the buffer
        self.register_buffer(name, value)
        self.saved_buffers.append(name)

    def forward(self):
        raise NotImplementedError("The forward method is not implemented.")

    ModuleTemplate = type(
        name,  # Class name
        bases,            # Base classes tuple
        {
            '__init__': __init__,
            'add_buffer': add_buffer,
            'forward': forward,
        }
    )

    return ModuleTemplate()

#########################################################
# Tagged datatype
#########################################################


def tag_tensor(tensor, tag):
    if hasattr(tensor, 'export_manager_tag'):
        tensor.export_manager_tag = tag
        return tensor

    tensor.export_manager_tag = tag
    tensor.value = lambda: tensor

    if not hasattr(torch.Tensor, '_original_clone'):
        def clone_with_tags(self, *args, **kwargs):
            """Clone tensor while preserving export_manager_tag and other custom attributes."""
            cloned = self._original_clone(*args, **kwargs)

            # Copy all custom attributes
            for attr_name in dir(self):
                if not attr_name.startswith('_') and not hasattr(torch.Tensor, attr_name):
                    if hasattr(self, attr_name):
                        setattr(cloned, attr_name, getattr(self, attr_name))

            return cloned

        # Monkey patch the clone method
        torch.Tensor._original_clone = torch.Tensor.clone
        torch.Tensor.clone = clone_with_tags

    return tensor


def tag_data(data, tag):
    '''
    Recursively expand data to tag underlying tensors in the data structure.
    if the data is already tagged, it will overwrite the tag.
    '''
    if isinstance(data, torch.Tensor):
        tag_tensor(data, tag)
    elif isinstance(data, collections.abc.Mapping):
        for key, value in data.items():
            tag_data(value, tag + "[" + key + "]")
    elif isinstance(data, collections.abc.Iterable) and not isinstance(data, (str, bytes)) and not hasattr(data, '__array__'):
        # This catches lists, tuples, sets, etc. but excludes strings, bytes, and numpy arrays
        for idx, item in enumerate(data):
            tag_data(item, tag + "[" + str(idx) + "]")
    else:
        print(
            f"\033[93mWarning: Untaggable datatype in i/o: {type(data)}\033[0m")


def get_tagged_subclass(base_class, value, tag):
    if tag is None:
        return value

    # special case for torch tensors:
    if isinstance(value, torch.Tensor):
        tag_tensor(value, tag)
        return value
    elif hasattr(base_class, '__module__') and base_class.__module__ == 'torch' and 'Tensor' in base_class.__name__:
        tag_tensor(value, tag)
        return value

    # For mutable types, try to preserve reference by adding attribute directly
    # This works for list, dict, set, bytearray, and other mutable collections
    if base_class in (list, set, bytearray) or hasattr(value, '__dict__'):
        try:
            value.export_manager_tag = tag
            # Add a method to get the original type (no lambda needed for identity)
            value.value = lambda: value
            return value  # Preserves original reference
        except (AttributeError, TypeError) as e:
            # Fall back to subclassing if direct attribute assignment fails
            pass

    def __new__(cls, value, tag=None):
        # bool is not subclassable; fall back to int while preserving bool semantics
        if base_class is bool:
            obj = int.__new__(cls, 1 if bool(value) else 0)
            obj.export_manager_tag = tag
            return obj
        obj = base_class.__new__(cls, value)
        obj.export_manager_tag = tag
        return obj

    def __init__(self, value, tag=None):
        # Ensure base __init__ doesn't see the 'export_manager_tag' kwarg (important for dict)
        if base_class is bool:
            # No-op; int.__init__ ignores args but don't forward export_manager_tag
            return
        try:
            base_class.__init__(self, value)
        except TypeError:
            # Some immutables ignore __init__ or have different signatures; safely ignore
            pass

        self.base_class = base_class

    def get_base_class(self):
        if base_class is bool:
            return bool(int(self))
        return self.base_class(self)

    # all other python base classes need to be wrapped in a TaggedBaseClass function
    new_cls = type(
        "Tagged" + base_class.__name__,
        (int,) if base_class is bool else (base_class,),
        {
            '__new__': __new__,
            '__init__': __init__,
            'value': get_base_class,
        }
    )

    new_class = new_cls(value, tag)

    return new_class
