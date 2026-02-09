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

import collections.abc
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple, List

import torch
import numpy as np
import yaml

from leapp._logging import _get_logger
from leapp.leapp_graph.datatypes import TracedData, TracedTensor
from leapp.utils.utils import safe_deepcopy
from leapp.utils.enums import inputKindEnum


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


# Register custom YAML representers so these types are always serialized in flow style
yaml.add_representer(
    CompactYamlList,
    lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True))
yaml.add_representer(
    CompactYamlDict,
    lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True))


class TensorDescription:
    """Describes a tensor input/output in the computational graph."""

    def __init__(self, name: str, value: Any, tag: Optional[str] = None,
                 kind: Optional[inputKindEnum] = None,
                 element_names: Optional[List[List[str]]] = None):
                 
        # extract tag from the input if not overridden
        if tag is None and hasattr(value, 'leapp_tag'):
            tag = value.leapp_tag

        # unwrap TracedData to the underlying tensor
        if isinstance(value, TracedData):
            value = value.tensor

        dtype, shape = TensorDescription.get_shape_and_dtype(value)

        self.name = name
        self.value = safe_deepcopy(value)
        self.dtype = dtype
        self.shape = CompactYamlList(shape)
        self.type = "tensor"
        self.tag = tag
        self.kind = kind
        self.element_names = element_names

    @staticmethod
    def get_shape_and_dtype(value) -> Tuple[str, tuple]:
        """Extract dtype string and shape from a torch.Tensor or numpy ndarray."""
        dtype_map = get_dtype_map()

        data_type = type(value)
        if data_type == torch.Tensor:
            dtype = dtype_map['torch'][value.dtype]
            shape = value.shape
        elif data_type == np.ndarray:
            dtype = dtype_map['numpy'][value.dtype]
            shape = value.shape
        else:
            raise ValueError(f"Unsupported type: {data_type}")

        return dtype, shape

    def dict(self, ignore_tag=True, ignore_value=True) -> Dict[str, Any]:
        """Convert to a dictionary."""
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
        if self.kind is not None:
            result["kind"] = self.kind.value
        if self.element_names is not None:
            result["element_names"] = self.element_names
        return result

    def change_name(self, new_name: str):
        """Change the name of the tensor description."""
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
    # TODO: use this to replace reference to the tensor descriptions.
    def tensor_expr_in_order(self) -> list[str]:
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
                items = [
                    f'"{k}": {_generate_expr(v)}' for k, v in format_item.items()]
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
                _generate_unpacking(format_item.formatting,
                                    accessor_path, assignments)
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

def get_dtype_map():
    return {
        "torch": {
            torch.float64: "float64",
            torch.float32: "float32",
            torch.float16: "float16",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.uint8: "uint8",
            torch.int8: "int8",
            torch.bool: "bool",
            torch.bfloat16: "bfloat16",
        },
        "numpy": {
            np.float64: "float64",
            np.float32: "float32",
            np.float16: "float16",
            np.int32: "int32",
            np.int64: "int64",
            np.uint8: "uint8",
            np.int8: "int8",
            np.bool_: "bool",
        },
    }

def map_to_torch_dtype(string):
    torch_map = get_dtype_map()['torch']
    reverse_map = {dtype_str: dtype for dtype, dtype_str in torch_map.items()}
    if string in reverse_map:
        return reverse_map[string]
    raise ValueError(f"Unsupported string: {string}")


def verify_data_exact_match(source_data, target_data):
    # Check if both are the same type (allow dict-like and list-like substitutions)
    source_is_list_like = isinstance(source_data, collections.abc.Sequence) and not isinstance(
        source_data, (str, bytes, torch.Tensor))
    target_is_list_like = isinstance(target_data, collections.abc.Sequence) and not isinstance(
        target_data, (str, bytes, torch.Tensor))
    source_is_dict_like = isinstance(source_data, collections.abc.Mapping)
    target_is_dict_like = isinstance(target_data, collections.abc.Mapping)
    if isinstance(source_data, TracedTensor):
        source_data = source_data.tensor
    if isinstance(target_data, TracedTensor):
        target_data = target_data.tensor

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


def flatten_io_structure(data, name_str):
    flat_data = {}
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes, torch.Tensor)):
        for idx, item in enumerate(data):
            child_name = f"{name_str}_{idx}" if name_str else str(idx)
            flat_data.update(flatten_io_structure(item, child_name))
    elif isinstance(data, collections.abc.Mapping):
        for key, value in data.items():
            child_name = f"{name_str}_{key}" if name_str else key
            flat_data.update(flatten_io_structure(value, child_name))
    elif isinstance(data, torch.Tensor) or isinstance(data, TracedTensor):
        flat_data[name_str] = data

    return flat_data


def describe_io_helper(data, name_str):
    data_description = []
    io_format = {}
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes, torch.Tensor)):
        if not isinstance(data, list):
            type_name = type(data).__name__
            _get_logger().warning(
                f"Input/Output '{name_str}' has list-like type '{type_name}' which will be "
                f"treated as 'list'. Ensure the runtime can handle this substitution.")

        io_format = []
        for idx, item in enumerate(data):
            child_name = f"{name_str}_{idx}" if name_str else str(idx)
            child_description, child_format = describe_io_helper(
                item, child_name)
            io_format.append(child_format)
            data_description.extend(child_description)
        return data_description, io_format
    elif isinstance(data, collections.abc.Mapping):
        if not isinstance(data, dict):
            type_name = type(data).__name__
            _get_logger().warning(
                f"Input/Output '{name_str}' has dict-like type '{type_name}' which will be "
                f"treated as 'dict'. Ensure the runtime can handle this substitution.")

        io_format = {}
        for k, v in data.items():
            child_name = f"{name_str}_{k}" if name_str else k
            child_description, child_format = describe_io_helper(
                v, child_name)
            io_format[k] = child_format
            data_description.extend(child_description)
    elif isinstance(data, torch.Tensor):
        tensor_desc = TensorDescription(name_str, data)

        # Return as a list containing the dataclass (for now, keep compatibility)
        data_description = [tensor_desc]
        io_format = tensor_desc  # Plain string
    
    elif isinstance(data, TensorDescription):
        # if the data is already a description we just return it
        data_description = [data]
        io_format = data

    else:
        _get_logger().error(f"Input/Output '{name_str}' has unsupported type: {type(data)}")

    return data_description, io_format


def describe_io(name, raw_name, data):
    data_description, io_format = describe_io_helper(
        data, name)
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
