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

import ast
import collections.abc
import copy
import inspect
import os
import sys
import textwrap
import traceback

import torch

from leapp._logging import _get_logger
from leapp.leapp_graph.datatypes import TracedData, TracedTensor

# Re-export from datatypes for backwards compatibility
from leapp.leapp_graph.datatypes import is_tracable_tensor_type  # noqa: F401

def find_with_block_end(filename, start_lineno):
    """Use AST to find the end line of the with block starting at start_lineno.
    
    This is deterministic and doesn't depend on runtime tracing behavior.
    
    Args:
        filename: Path to the source file
        start_lineno: Line number where the 'with' statement starts
        
    Returns:
        The end line number of the with block, or None if not found
    """
    with open(filename, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            if node.lineno == start_lineno:
                return node.end_lineno
    return None


def extract_source_from_line_range(executed_lines, context_name, is_function=False):
    """
    Extract source code from a traced line range.

    This is the unified source extraction function used by both function decorators
    and block contexts. It uses AST to properly determine body boundaries and handles
    multiline statements (like dict comprehensions) that only trigger a single line event.

    Args:
        executed_lines: Dictionary with keys 'filename', 'min_line', 'max_line', 'function_name'
        context_name: Name of the context/node for logging purposes
        is_function: If True, look for FunctionDef/AsyncFunctionDef. If False, look for With/AsyncWith.

    Returns:
        tuple: (source_code, message) where source_code is the extracted code or empty string
    """
    if executed_lines['filename'] is None:
        return "", f"No filename provided for {context_name}"

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
            return "", (
                f"Warning {context_name}: Line range {min_line}-{max_line} "
                f"is out of bounds for file {filename}"
            )

        # Determine the start of the executable body using AST (robust for multiline headers)
        file_source = ''.join(source_lines)
        body_start_idx = None

        try:
            tree = ast.parse(file_source)

            best_candidate = None
            best_start = -1

            # Choose AST node types based on whether this is a function or block context
            if is_function:
                wanted = (ast.FunctionDef, ast.AsyncFunctionDef)
            else:
                wanted = (ast.With, ast.AsyncWith)

            # Find the node that contains our line range
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

                    # Check if this node overlaps with our line range
                    # (handles decorators by checking if def line or any body line is in range)
                    if (min_line <= node_start <= max_line) or (node_start <= min_line <= node_end):
                        if node_start > best_start:
                            best_candidate = n
                            best_start = node_start

            if best_candidate is not None and getattr(best_candidate, 'body', None):
                body_start_line = getattr(
                    best_candidate.body[0], 'lineno', None)
                if body_start_line is not None:
                    body_start_idx = body_start_line - 1

                # Extend end_idx to capture multiline statements
                # Python's line tracer only fires once for multiline statements (e.g., dict comprehensions),
                # so max_line may not include the closing brace. Use AST's end_lineno instead.
                ast_end_line = getattr(best_candidate, 'end_lineno', None)
                if ast_end_line is not None and ast_end_line > end_idx:
                    end_idx = ast_end_line
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
            _get_logger().warning(f"Warning: Function {func.__name__} has {len(return_statements)} return statements. "
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


def safe_deepcopy(data):
    # this is used to deepcopy a complex data structure.
    # running safe deepcopy also unwraps the TracedTensor to the underlying tensor.
    if isinstance(data, torch.Tensor):
        cloned_data = data.clone()
        if hasattr(data, 'leapp_tag'):
            tag_tensor(cloned_data, data.leapp_tag)
        return cloned_data
    elif isinstance(data, list):
        return [safe_deepcopy(item) for item in data]
    elif isinstance(data, dict):
        return {k: safe_deepcopy(v) for k, v in data.items()}
    else:
        return copy.deepcopy(data)


def frame_to_namespace(frame):
    """Convert a Python frame object to a namespace dictionary.
    
    Args:
        frame: A Python frame object from sys._getframe() or similar
        
    Returns:
        dict: A namespace combining globals and locals from the frame
    """
    return {**frame.f_globals, **frame.f_locals}


def get_attribute_value_from_namespace(namespace, attr_name):
    """Look up a variable (possibly dotted like 'self.counter') from a namespace dict.
    
    Args:
        namespace: dict containing variables to look up
        attr_name: Variable name, possibly dotted (e.g., 'self.counter', 'my_var')
    
    Returns:
        tuple: (value, final_attr_name) where final_attr_name is the last part of dotted name
    """
    if "." in attr_name:
        parts = attr_name.split(".")
        final_attr_name = parts[-1]
        if parts[0] not in namespace:
            raise Exception(f"Variable '{parts[0]}' not found in namespace")
        
        obj = namespace[parts[0]]
        for attr in parts[1:]:
            try:
                obj = getattr(obj, attr)
            except Exception as e:
                raise Exception(
                    f"Error attempting to find {attr_name}, "
                    f"failed to get attribute {attr}\n", e)
    else:
        if attr_name not in namespace:
            raise Exception(f"Variable '{attr_name}' not found in namespace")
        obj = namespace[attr_name]
        final_attr_name = attr_name

    return obj, final_attr_name


#########################################################
# Tagged datatype
#########################################################


def tag_tensor(tensor, tag):
    """Tag a tensor instance with a leapp_tag and wrap methods to preserve the tag.
    
    This binds wrapper methods to the specific tensor INSTANCE only.
    Works with both torch.Tensor and numpy arrays (including TracedNpArray).
    """
    if hasattr(tensor, 'leapp_tag'):
        # Already tagged - just update the tag
        tensor.leapp_tag = tag
        return tensor

    tensor.leapp_tag = tag

    # Methods to wrap depend on tensor type
    # torch.Tensor: clone, detach, contiguous, cpu, cuda, to
    # numpy.ndarray: copy (numpy's equivalent of clone)
    if isinstance(tensor, torch.Tensor):
        methods_to_wrap = ['clone', 'detach', 'contiguous', 'cpu', 'cuda', 'to']
    else:
        # For numpy arrays (including TracedNpArray), only wrap methods that exist
        methods_to_wrap = ['copy']  # numpy's equivalent of clone

    for method_name in methods_to_wrap:
        # Only wrap if the method exists on the tensor
        if not hasattr(tensor, method_name):
            continue
        # Store the original bound method on the instance
        original_attr = f'_original_{method_name}'
        if not hasattr(tensor, original_attr):
            setattr(tensor, original_attr, getattr(tensor, method_name))

        def make_wrapper(mname, source_tensor):
            """Create a wrapper that preserves leapp_tag on the result tensor."""
            orig_attr = f'_original_{mname}'

            def wrapper(*args, **kwargs):
                original_method = getattr(source_tensor, orig_attr)
                result = original_method(*args, **kwargs)
                # Tag the result tensor with the same tag (and wrap its methods too)
                if hasattr(source_tensor, 'leapp_tag'):
                    tag_tensor(result, source_tensor.leapp_tag)
                return result

            return wrapper

        # Bind wrapper to this specific instance (shadows the class method)
        setattr(tensor, method_name, make_wrapper(method_name, tensor))

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


_LEAPP_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_caller_stack_identity():
    """Return a hashable tuple representing the caller's full stack trace.

    Captures only frames outside the leapp package so that two calls reaching
    the same leapp API through different user-code paths produce distinct
    identities (e.g. user helper wrappers around input_tensors).
    """
    return tuple(
        (f.filename, f.lineno, f.name) for f in traceback.extract_stack()
        if not f.filename.startswith(_LEAPP_PKG_DIR)
    )


def format_caller_identity(identity):
    """Pretty-print a caller identity for error messages."""
    if not isinstance(identity, tuple) or not identity:
        return str(identity)
    if len(identity) == 2 and isinstance(identity[0], str) and isinstance(identity[1], int):
        return f"{identity[0]}:{identity[1]}"
    lines = []
    for frame in identity:
        if isinstance(frame, tuple) and len(frame) == 3:
            lines.append(f"  {frame[0]}:{frame[1]} in {frame[2]}")
        else:
            lines.append(f"  {frame}")
    return "\n".join(lines)
