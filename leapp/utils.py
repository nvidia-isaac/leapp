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
import os
import sys
import collections.abc
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple
from leapp.leapp_graph.traced_tensor import TracedTensor


def extract_source_from_line_range(executed_lines, context_name):
    """
    Extract source code from a traced line range.

    Args:
        executed_lines: Dictionary with keys 'filename', 'min_line', 'max_line', 'function_name'
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

            best_candidate = None
            best_start = -1

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
        message = f"Extracted code from file: {filename}, function: {executed_lines['function_name']}"

    except Exception as e:
        message = f"Error extracting source from line range: {e}"
        source_code = ""

    return source_code, message


class CompactYamlList(list):
    """Custom list class for tensor shapes that enables compact YAML formatting
    This class will behave exactly like a list. The only difference is that it will be
    formatted in a more compact way in the yaml file.

    """
    pass


class CompactYamlDict(dict):
    """Custom dict class for tensor shapes that enables compact YAML formatting
    This class will behave exactly like a dict. The only difference is that it will be
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


@dataclass
class ParameterFormat:
    name: str
    formatting: Any

    def get_format_string(self, data: Any) -> Tuple[bool, str]:

        if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes, torch.Tensor)):
            # Warn if not a native list
            if not isinstance(data, list):
                type_name = type(data).__name__
                print(
                    f"Parameter '{self.name}' has list-like type '{type_name}' which will be "
                    f"treated as 'list' in the generated function signature. This may cause "
                    f"issues if the exact type is required.")

            if len(data) == 0:
                return True, "list"  # Empty list, can't determine child type

            item_types_valid = len(set(type(item) for item in data)) == 1
            if not item_types_valid:
                return False, "INCONSISTENT LIST ITEM TYPES"
            else:
                success, child = self.get_format_string(data[0])
                if not success:
                    return False, "LIST CHILD FORMAT STRING NOT VALID"
                else:
                    return True, f"List[{child}]"

        elif isinstance(data, collections.abc.Mapping):
            # Warn if not a native dict
            if not isinstance(data, dict):
                type_name = type(data).__name__
                print(
                    f"Parameter '{self.name}' has dict-like type '{type_name}' which will be "
                    f"treated as 'dict' in the generated function signature. This may cause "
                    f"issues if the exact type is required.")

            if len(data) == 0:
                return True, "dict"  # Empty dict, can't determine child type

            key_types = set(type(key) for key in data.keys())
            if len(key_types) > 1 or key_types != {str}:
                return False, "INCONSISTENT DICT KEY TYPES"
            value_types = set(type(value) for value in data.values())
            if len(value_types) > 1:
                return False, "INCONSISTENT DICT VALUE TYPES"

            success, child = self.get_format_string(
                list(data.values())[0])
            if not success:
                return False, "DICT CHILD FORMAT STRING NOT VALID"
            else:
                return True, f"Dict[str, {child}]"

        elif isinstance(data, TensorDescription):
            return True, type(data.value).__name__

        else:
            return False, "UNEXPECTED TYPE"

    @property
    def name_str(self) -> str:
        """Return just the name as a string. 
        will do some modifications to me the vairable compliant with python"""
        return self.name.replace(".", "_")

    @property
    def name_raw(self) -> str:
        return self.name

    @property
    def dtype(self) -> str:
        """Return the dtype as a string."""
        success, dtype = self.get_format_string(self.formatting)
        if not success:
            return None
        return dtype

    @property
    def valid(self):
        a = self.get_format_string(self.formatting)
        if not a:
            return False
        return True

    @property
    def tensor_expr_in_order(self) -> list[str]: #TODO: use this to replace reference to the tensor descriptions.
        """
        Generate a list of tensor expressions in the order of the nested structure.
        """
        def _generate_expr(format_item) -> list[str]:
            if isinstance(format_item, ParameterFormat):
                return _generate_expr(format_item.formatting)
            elif isinstance(format_item, TensorDescription):
                return [format_item.name_str]
            elif isinstance(format_item, list):
                return [_generate_expr(item) for item in format_item]
            elif isinstance(format_item, dict):
                return [_generate_expr(value) for value in format_item.values()]
            else:
                return []
        return _generate_expr(self.formatting)

    @property
    def packed_tensor_expr(self) -> str:
        """
        Generate a packing assignment expression for this parameter.
        
        Converts flat tensor inputs into the expected nested structure.
        Handles nested structures (lists, dicts).
        
        Returns:
            Assignment string like "inputA = [inputA_0, inputA_1]". 
            Empty string if no packing needed.
        """
        def _generate_expr(format_item) -> str:
            if isinstance(format_item, ParameterFormat):
                return _generate_expr(format_item.formatting)
            elif isinstance(format_item, TensorDescription):
                return format_item.name_str
            elif isinstance(format_item, list):
                elements = [_generate_expr(item) for item in format_item]
                return "[" + ", ".join(elements) + "]"
            elif isinstance(format_item, dict):
                items = [f'"{k}": {_generate_expr(v)}' for k, v in format_item.items()]
                return "{" + ", ".join(items) + "}"
            else:
                return "None"
        
        reconstruction = _generate_expr(self.formatting)
        
        # Skip trivial assignments where name == expression
        if self.name_str == reconstruction:
            return ""
        
        return f"{self.name_str} = {reconstruction}"

    @property
    def unpacked_tensor_expr(self) -> str:
        """
        Generate unpacking assignment expressions that extract flat tensors from
        a nested structure, using accessor paths.
        
        This is the inverse of packed_tensor_expr.
        
        Handles:
        - List unpacking (e.g., "inputA_0 = inputA[0]", "inputA_1 = inputA[1]")
        - Dict unpacking (e.g., 'state_pose = state["pose"]')
        - Nested structures (e.g., 'nested_0_0 = nested[0][0]')
        
        Returns:
            Newline-separated assignment strings. Empty string if no unpacking needed.
        """
        def _generate_unpacking(format_item, accessor_path: str, assignments: list):
            if isinstance(format_item, ParameterFormat):
                _generate_unpacking(format_item.formatting, accessor_path, assignments)
            elif isinstance(format_item, TensorDescription):
                tensor_name = format_item.name_str
                if tensor_name != accessor_path:
                    assignments.append(f"{tensor_name} = {accessor_path}")
            elif isinstance(format_item, list):
                for idx, item in enumerate(format_item):
                    child_path = f"{accessor_path}[{idx}]"
                    _generate_unpacking(item, child_path, assignments)
            elif isinstance(format_item, dict):
                for key, value in format_item.items():
                    child_path = f'{accessor_path}["{key}"]'
                    _generate_unpacking(value, child_path, assignments)
        
        assignments = []
        _generate_unpacking(self.formatting, self.name_str, assignments)
        return "\n".join(assignments)


def verify_data_exact_match(source_data, target_data):
    # Check if both are the same type (allow dict-like and list-like substitutions)
    source_is_list_like = isinstance(source_data, collections.abc.Sequence) and not isinstance(
        source_data, (str, bytes, torch.Tensor))
    target_is_list_like = isinstance(target_data, collections.abc.Sequence) and not isinstance(
        target_data, (str, bytes, torch.Tensor))
    source_is_dict_like = isinstance(source_data, collections.abc.Mapping)
    target_is_dict_like = isinstance(target_data, collections.abc.Mapping)

    # If one is list-like and the other is not, they don't match
    if source_is_list_like != target_is_list_like:
        return False
    # If one is dict-like and the other is not, they don't match
    if source_is_dict_like != target_is_dict_like:
        return False

    if isinstance(source_data, torch.Tensor):
        if not isinstance(target_data, torch.Tensor):
            return False
        if source_data.shape != target_data.shape:
            return False
        if source_data.dtype != target_data.dtype:
            return False
        if source_data.device != target_data.device:
            return False
        if not torch.equal(source_data, target_data):
            return False

    elif source_is_list_like:
        if len(source_data) != len(target_data):
            return False
        for source_item, target_item in zip(source_data, target_data):
            if not verify_data_exact_match(source_item, target_item):
                return False
    elif source_is_dict_like:
        if set(source_data.keys()) != set(target_data.keys()):
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
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes, torch.Tensor)):
        if not isinstance(data, list):
            type_name = type(data).__name__
            print(
                f"Input/Output '{name_str}' has list-like type '{type_name}' which will be "
                f"treated as 'list'. Ensure the runtime can handle this substitution.")

        io_format = []
        for idx, item in enumerate(data):
            child_name = "_".join([name_str, str(idx)])
            child_description, child_format = describe_io_helper(
                item, child_name, dtype_notation)
            io_format.append(child_format)
            data_description.extend(child_description)
        return data_description, io_format
    elif isinstance(data, collections.abc.Mapping):
        if not isinstance(data, dict):
            type_name = type(data).__name__
            print(
                f"Input/Output '{name_str}' has dict-like type '{type_name}' which will be "
                f"treated as 'dict'. Ensure the runtime can handle this substitution.")

        io_format = {}
        for k, v in data.items():
            child_name = "_".join([name_str, k])
            child_description, child_format = describe_io_helper(
                v, child_name, dtype_notation)
            io_format[k] = child_format
            data_description.extend(child_description)
    elif isinstance(data, torch.Tensor):
        tag = None
        if hasattr(data, 'leapp_tag'):
            tag = data.leapp_tag

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


def describe_io(name, raw_name, data, dtype_notation="python"):
    data_description, io_format = describe_io_helper(
        data, name, dtype_notation)
    parameter_description = ParameterFormat(
        name=raw_name, formatting=io_format)
    return data_description, parameter_description


def _resolve_tensor_descriptions(io_format, extractor):
    """
    Recursively resolve TensorDescription objects using a custom extractor function.

    This function traverses through nested structures (lists, dicts, etc.) and
    applies the extractor function to any TensorDescription objects found.

    Args:
        io_format: The object to recursively process (can be TensorDescription, ParameterFormat, list, dict, or any other type)
        extractor: A function that takes a TensorDescription and returns the desired value

    Returns:
        The processed object with TensorDescription objects replaced by extracted values
    """
    if isinstance(io_format, ParameterFormat):
        # For ParameterFormat, apply resolution to the formatting attribute
        return _resolve_tensor_descriptions(io_format.formatting, extractor)
    elif isinstance(io_format, TensorDescription):
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
        io_format: The object to recursively process (can be ParameterFormat, TensorDescription, list, dict, etc.)

    Returns:
        The processed object with TensorDescription objects replaced by their string names.
        If a ParameterFormat is passed, returns the resolved formatting from within it.
    """
    return _resolve_tensor_descriptions(io_format, lambda td: td.name_str)


def resolve_tensor_descriptions_to_values(io_format):
    """
    Recursively resolve TensorDescription objects to their values.

    Args:
        io_format: The object to recursively process (can be ParameterFormat, TensorDescription, list, dict, etc.)

    Returns:
        The processed object with TensorDescription objects replaced by their values.
        If a ParameterFormat is passed, returns the resolved formatting from within it.
    """
    return _resolve_tensor_descriptions(io_format, lambda td: td.value)


def reconstruct_from_named_dict(named_dict, io_format, use_tag_first=True):
    """
    Reverse of describe_io_helper: reconstruct data structure from named dict and format.

    Takes a dictionary mapping names/tags to actual data values and an io_format structure,
    and reconstructs the original nested data structure.

    Args:
        named_dict: Dictionary mapping names or tags to actual data values
        io_format: The format structure (can be ParameterFormat, TensorDescription objects or strings)
        use_tag_first: If True, try to lookup by tag first, then by name. If False, use name only.

    Returns:
        The reconstructed data structure with the same shape as the original

    Example:
        # Original data
        data = {'a': tensor1, 'b': [tensor2, tensor3]}

        # Describe it
        desc, fmt = describe_io_helper(data, "root", "python")

        # Later reconstruct from a named dict
        named = {'root.a': new_tensor1, 'root.b.0': new_tensor2, 'root.b.1': new_tensor3}
        reconstructed = reconstruct_from_named_dict(named, fmt)
        # reconstructed = {'a': new_tensor1, 'b': [new_tensor2, new_tensor3]}
    """
    def resolve(item):
        if isinstance(item, ParameterFormat):
            # For ParameterFormat, extract the formatting attribute
            return resolve(item.formatting)
        elif isinstance(item, TensorDescription):
            # Try to find the value in named_dict
            # First try by tag if it exists
            if item.tag is not None and item.tag in named_dict:
                return named_dict[item.tag]

            # Then try by name
            if item.name in named_dict:
                return named_dict[item.name]

            raise KeyError(
                f"Could not find data for TensorDescription with name='{item.name}' and tag='{item.tag}' in named_dict")
        elif isinstance(item, str):
            # If it's a string, look it up directly in named_dict
            if item in named_dict:
                return named_dict[item]
            raise KeyError(
                f"Could not find data for key '{item}' in named_dict")
        elif isinstance(item, collections.abc.Sequence) and not isinstance(item, (str, bytes, torch.Tensor)):
            return [resolve(sub_item) for sub_item in item]
        elif isinstance(item, collections.abc.Mapping):
            return {key: resolve(value) for key, value in item.items()}
        else:
            # Return as-is for other types
            return item

    return resolve(io_format)


def flatten_to_named_dict(data, io_format, use_tag_first=True):
    """
    Flatten a nested data structure to a dictionary using io_format as a guide.

    This is the inverse of reconstruct_from_named_dict: given a nested data structure
    and its corresponding format, it creates a flat dictionary mapping tags/names to values.

    Args:
        data: The nested data structure (can be tensor, list, dict, etc.)
        io_format: The format structure (can be ParameterFormat or containing TensorDescription objects)
        use_tag_first: If True, use tag as key if available, otherwise use name

    Returns:
        Dictionary mapping tags/names to actual values

    Example:
        # Original data
        data = {'a': tensor1, 'b': [tensor2, tensor3]}

        # Describe it to get format
        desc, fmt = describe_io_helper(data, "root", "python")

        # Flatten it back
        flat = flatten_to_named_dict(data, fmt)
        # flat = {'root.a': tensor1, 'root.b.0': tensor2, 'root.b.1': tensor3}
    """
    result = {}

    def flatten(data_item, format_item):
        if isinstance(format_item, ParameterFormat):
            # For ParameterFormat, extract the formatting attribute
            flatten(data_item, format_item.formatting)
        elif isinstance(format_item, TensorDescription):
            # Extract the key (tag or name)
            if use_tag_first and format_item.tag is not None:
                key = format_item.tag
            else:
                key = format_item.name

            # Store the value
            result[key] = data_item

        elif isinstance(format_item, collections.abc.Sequence) and not isinstance(format_item, (str, bytes, torch.Tensor)):
            # Both should be lists or list-like
            if not (isinstance(data_item, collections.abc.Sequence) and not isinstance(data_item, (str, bytes, torch.Tensor))):
                raise TypeError(
                    f"Format expects a list but data is {type(data_item)}")
            if len(data_item) != len(format_item):
                raise ValueError(
                    f"List length mismatch: data has {len(data_item)} items, "
                    f"format expects {len(format_item)}")

            for data_sub, format_sub in zip(data_item, format_item):
                flatten(data_sub, format_sub)

        elif isinstance(format_item, collections.abc.Mapping):
            # Both should be dicts or dict-like
            if not isinstance(data_item, collections.abc.Mapping):
                raise TypeError(
                    f"Format expects a dict but data is {type(data_item)}")
            if set(data_item.keys()) != set(format_item.keys()):
                raise ValueError(
                    f"Dict keys mismatch: data has {set(data_item.keys())}, "
                    f"format expects {set(format_item.keys())}")

            for key in format_item.keys():
                flatten(data_item[key], format_item[key])
        else:
            # For other types, we don't have a good way to extract a key
            # This shouldn't normally happen if format was created by describe_io_helper
            pass

    flatten(data, io_format)
    return result


def safe_deepcopy(data):
    # this is used to deepcopy a complex data structure.
    # running safe deepcopy also unwraps the TracedTensor to the underlying tensor.
    if isinstance(data, torch.Tensor):
        return data.clone()
    elif isinstance(data, TracedTensor):
        return data.tensor.clone()

    elif isinstance(data, list):
        return [safe_deepcopy(item) for item in data]
    elif isinstance(data, dict):
        return {k: safe_deepcopy(v) for k, v in data.items()}
    else:
        return copy.deepcopy(data)

def get_attribute_value_from_frame(frame, attr_name):
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
# Tagged datatype
#########################################################


def tag_tensor(tensor, tag):
    if hasattr(tensor, 'leapp_tag'):
        tensor.leapp_tag = tag
        return tensor

    tensor.leapp_tag = tag
    tensor.value = lambda: tensor

    # Helper function to copy custom attributes from source to target tensor
    def _copy_custom_attrs(source, target):
        """Copy all custom attributes from source tensor to target tensor."""
        for attr_name in dir(source):
            if not attr_name.startswith('_') and not hasattr(torch.Tensor, attr_name):
                if hasattr(source, attr_name):
                    setattr(target, attr_name, getattr(source, attr_name))
        return target

    # Helper function to create a monkey-patched method wrapper
    def _make_wrapper(method_name, docstring):
        """Create a wrapper function that preserves custom attributes."""
        original_attr = f'_original_{method_name}'

        def wrapper(self, *args, **kwargs):
            original_method = getattr(self, original_attr)
            result = original_method(*args, **kwargs)
            return _copy_custom_attrs(self, result)

        wrapper.__doc__ = docstring
        return wrapper

    # List of methods to monkey patch
    methods_to_patch = [
        ('clone', 'Clone tensor while preserving leapp_tag and other custom attributes.'),
        ('detach', 'Detach tensor while preserving leapp_tag and other custom attributes.'),
        ('contiguous', 'Make tensor contiguous while preserving leapp_tag and other custom attributes.'),
        ('cpu', 'Move tensor to CPU while preserving leapp_tag and other custom attributes.'),
        ('cuda', 'Move tensor to CUDA while preserving leapp_tag and other custom attributes.'),
    ]

    # Apply monkey patches
    for method_name, docstring in methods_to_patch:
        original_attr = f'_original_{method_name}'
        if not hasattr(torch.Tensor, original_attr):
            # Save original method
            setattr(torch.Tensor, original_attr,
                    getattr(torch.Tensor, method_name))
            # Replace with wrapper
            setattr(torch.Tensor, method_name,
                    _make_wrapper(method_name, docstring))
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


def mirror_all_tensor_tags(source, target):
    '''
    Mirror all tensor tags from source to target.
    '''
    if isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor):
        if hasattr(source, 'leapp_tag'):
            tag_tensor(target, source.leapp_tag)
    elif isinstance(source, collections.abc.Mapping):
        for key, value in source.items():
            mirror_all_tensor_tags(value, target[key])
    elif isinstance(source, collections.abc.Iterable):
        for idx, item in enumerate(source):
            mirror_all_tensor_tags(item, target[idx])


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
            value.leapp_tag = tag
            # Add a method to get the original type (no lambda needed for identity)
            value.value = lambda: value
            return value  # Preserves original reference
        except (AttributeError, TypeError):
            # Fall back to subclassing if direct attribute assignment fails
            pass

    def __new__(cls, value, tag=None):
        # bool is not subclassable; fall back to int while preserving bool semantics
        if base_class is bool:
            obj = int.__new__(cls, 1 if bool(value) else 0)
            obj.leapp_tag = tag
            return obj
        obj = base_class.__new__(cls, value)
        obj.leapp_tag = tag
        return obj

    def __init__(self, value, tag=None):
        # Ensure base __init__ doesn't see the 'leapp_tag' kwarg (important for dict)
        if base_class is bool:
            # No-op; int.__init__ ignores args but don't forward leapp_tag
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


def get_relative_path(model_path, yaml_save_path):
    """
    Convert a single model path to be relative to the YAML file location.

    Args:
        model_path: Absolute path to the model file
        yaml_save_path: Directory path where the YAML file will be saved

    Returns:
        Relative model path, or original path if conversion fails
    """
    if not yaml_save_path or not model_path:
        return model_path

    try:
        # Convert to relative path
        return os.path.relpath(model_path, yaml_save_path)
    except ValueError:
        # If relative path calculation fails (e.g., different drives on Windows), keep absolute path
        return model_path


def get_system_info():
    import leapp
    metadata = {'system information': {}}
    metadata['system information']['leapp version'] = leapp.__version__
    metadata['system information']['torch version'] = str(torch.__version__)
    metadata['system information']['python version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    metadata['system information']['cuda version'] = str(torch.version.cuda)
    metadata['system information']['os'] = os.uname().sysname

    return metadata
