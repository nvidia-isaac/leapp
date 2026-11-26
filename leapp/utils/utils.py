#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ast
import collections.abc
import copy
import inspect
import os
import platform
import sys
import textwrap

import torch

from leapp.utils.logging import _get_logger
from leapp.leapp_graph.datatypes import (  # noqa: F401
    is_tracable_tensor_type,
    is_traced_type,
    promote_in_place,
    to_export_torch_tensor,
)


def extract_return_names(func):
    """Extract variable names from return statements in the function."""
    try:
        # Unwrap decorated functions to get to the original source
        unwrapped_func = inspect.unwrap(func)
        source = inspect.getsource(unwrapped_func)
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
                            statement_names.append(f"output{i + 1}")
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
            count_details = {
                count: return_counts.count(count) for count in unique_counts
            }
            _get_logger().fatal(
                f"Function {func.__name__} has inconsistent return statements with different numbers of values: "
                f"{count_details}. All return statements must return the same number of values.",
                error_type=Exception,
            )

        # All return statements have the same number of values
        num_returns = return_counts[0]

        if len(return_statements) > 1:
            # Multiple return statements - log warning and intelligently merge
            _get_logger().warning(
                f"Warning: Function {func.__name__} has {len(return_statements)} return statements. "
                f"This may cause unexpected behavior if the return statements return different "
                f"data types, shapes, or dtypes."
            )

        # Intelligently merge return names by position, prioritizing named over generic
        final_names = []
        for pos in range(num_returns):
            # Collect names at this position from all return statements
            names_at_position = [stmt[pos] for stmt in return_statements]

            # Separate named variables from generic ones
            named_vars = [
                name for name in names_at_position if not name.startswith("output")
            ]
            generic_vars = [
                name for name in names_at_position if name.startswith("output")
            ]

            # Choose the best name for this position
            if named_vars:
                # Prefer named variables, take the first unique one
                chosen_name = named_vars[0]
            elif generic_vars:
                # Only generic names available
                chosen_name = generic_vars[0]
            else:
                # Fallback
                chosen_name = f"output{pos + 1}"

            final_names.append(chosen_name)

        return final_names

    except Exception as e:
        # Fallback to generic names if parsing fails
        _get_logger().error(
            f"Error extracting return names from {func.__name__}, reverting to defaults: {e}"
        )
        return ["output1"]


def safe_deepcopy(data):
    # this is used to deepcopy a complex data structure.
    # running safe deepcopy also unwraps the TracedTensor to the underlying tensor.
    if is_tracable_tensor_type(data):
        return to_export_torch_tensor(data).clone()
    elif isinstance(data, list):
        return [safe_deepcopy(item) for item in data]
    elif isinstance(data, dict):
        return {k: safe_deepcopy(v) for k, v in data.items()}
    else:
        return copy.deepcopy(data)


#########################################################
# Traced state transfer
#########################################################


def mirror_all_traced_state(source, target):
    """
    Copy the traced boundary state of every tensor in source onto target.

    Torch and Warp targets are upgraded in place, so the caller's object is
    updated directly. A raw numpy target becomes a zero-copy view, which is a
    new object, so the rebuilt target is returned and numpy callers must use it.
    """
    if is_tracable_tensor_type(source) and is_tracable_tensor_type(target):
        if not is_traced_type(source):
            _get_logger().error(
                f"Error: cannot mirror traced state from a plain {type(source).__name__}.\n"
                "Only values registered as a node output carry the state needed "
                "to connect nodes."
            )
            return target
        promoted = promote_in_place(
            target, source.name, source.context_obj, source.proxy
        )
        promoted.output_port = source.output_port
        return promoted
    elif isinstance(source, collections.abc.Mapping):
        return type(target)(
            (key, mirror_all_traced_state(value, target[key]))
            for key, value in source.items()
        )
    elif isinstance(source, collections.abc.Iterable):
        return type(target)(
            mirror_all_traced_state(item, target[idx])
            for idx, item in enumerate(source)
        )
    return target


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


def _get_os_description():
    freedesktop_os_release = getattr(platform, "freedesktop_os_release", None)
    if freedesktop_os_release is None:
        return platform.platform()

    try:
        os_release = freedesktop_os_release()
    except OSError:
        return platform.platform()

    name = os_release.get("NAME")
    version = os_release.get("VERSION_ID") or os_release.get("VERSION")
    if name and version:
        return f"{name} {version}"

    return os_release.get("PRETTY_NAME") or platform.platform()


def _get_warp_version():
    try:
        import warp
    except ImportError:
        return None

    return str(warp.__version__)


def get_system_info():
    # lazy import to avoid circular import
    import leapp

    metadata = {"system information": {}}
    metadata["system information"]["leapp version"] = leapp.__version__
    metadata["system information"]["leapp config version"] = leapp.__config_version__
    metadata["system information"]["torch version"] = str(torch.__version__)
    metadata["system information"]["python version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    metadata["system information"]["cuda version"] = str(torch.version.cuda)
    metadata["system information"]["warp version"] = _get_warp_version()
    metadata["system information"]["os"] = _get_os_description()

    return metadata
