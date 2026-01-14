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

import sys
import functools
import inspect
import yaml
import os
import torch
from typing import Any

from leapp._logging import _get_logger
from leapp.leapp_graph.leapp_graph import LeappGraph
from leapp.leapp_graph.function_decorator_node import FunctionDecoratorNode
from leapp.leapp_graph.leapp_node import LeappNode
from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.traced_tensor import TracedTensor
from leapp.leapp_graph.block_context_node import BlockContextNode
from leapp.utils import frame_to_namespace
from leapp.enums import MergeCfgEnum
from leapp.tracing_lock import TracingLock

from .utils import (CompactYamlList,
                    CompactYamlDict,
                    find_with_block_end,
                    get_relative_path,
                    get_system_info,
                    verify_data_exact_match,
                    mirror_all_tensor_tags,
                    flatten_io_structure)


class ExportManager:
    _instance = None
    _initialized = False  # True after singleton __init__ completes
    # True between start() and stop() - enables graph interpretation
    _interpret_graph = False

    #########################################################
    # initialization
    #########################################################

    def __new__(cls, *args, **kwargs):
        """Singleton implementation - only one instance allowed."""
        if cls._instance is None:
            cls._instance = super(ExportManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ExportManager._initialized:
            # graph settings
            self.GRAPH_NAME = "my_graph"
            self.SAVE_PATH = None

            # tracetime variables
            self.nodes = {}

            # Set up custom YAML representers before writing any YAML
            def represent_shape_list(dumper, data):
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

            def represent_shape_dict(dumper, data):
                return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)

            # Register the custom representers
            yaml.add_representer(CompactYamlList, represent_shape_list)
            yaml.add_representer(CompactYamlDict, represent_shape_dict)

            ExportManager._initialized = True

    #########################################################
    # flow control
    #########################################################
    def start(self, name, save_path=".", verbose=False):
        """Initialize and start LEAPP graph interpretation.

        This method prepares the export manager for tracing by setting up the graph name,
        creating the save directory, configuring the logger, and enabling graph interpretation.
        If graph interpretation is already active, it will reset the current graph state.

        Args:
            name (str): The name of the graph to be created. This will be used as the 
                directory name where graph artifacts are saved.
            save_path (str, optional): The base directory path where the graph directory 
                will be created. Defaults to "." (current directory).
            verbose (bool, optional): If True, enables verbose logging output. 
                Defaults to False.

        Returns:
            None

        Note:
            - The full save path will be: {save_path}/{name}/
            - Calling start() while interpretation is already active will reset the graph. This is discouraged.
        """
        self.GRAPH_NAME = name
        self.SAVE_PATH = os.path.join(save_path, self.GRAPH_NAME)
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        _get_logger().configure(self.SAVE_PATH, verbose=verbose)
        if ExportManager._interpret_graph:
            _get_logger().warning("LEAPP graph interpretation is already enabled, "
                                  "calling start() again will reset the graph")
            _get_logger().warning("Resetting graph...")
            TracingLock().reset()
        self.nodes = {}
        ExportManager._interpret_graph = True

    def stop(self):
        """Stop LEAPP graph interpretation and disable tracing.

        This method disables graph interpretation mode that was previously enabled by start().
        It performs safety checks to ensure that no active tracing is in progress and that
        graph interpretation is currently enabled before stopping.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: If ExportManager is currently in the middle of tracing a node.
            Exception: If graph interpretation is not currently enabled.

        Note:
            - This method should only be called after start() has been called.
            - Ensure all active tracing operations are completed before calling stop().
        """
        if TracingLock().is_active:
            raise Exception("ExportManager is currently tracing")
        if not ExportManager._interpret_graph:
            raise Exception("ExportManager graph interpretation is disabled")
        ExportManager._interpret_graph = False

    #########################################################
    # node setup
    #########################################################
    def get_node_index(self, name):
        if name in self.nodes.keys():
            # retracing inherits the node index of the original node
            node_index = self.nodes[name].node_index
        else:
            node_index = len(self.nodes)
        return node_index

    def _verify_no_active_function_tracing(self):
        if TracingLock().is_active:
            _get_logger().error(
                "Error when attempting to set up new trace\n"
                "ExportManager is already tracing")
            raise Exception("Error when attempting to set up new trace")

    def _setup_new_node(self, name, node_class: LeappNode, **kwargs):
        self._verify_no_active_function_tracing()
        node_index = self.get_node_index(name)

        node = node_class(name, node_index,
                          backend=kwargs.get("export_with", None),
                          backend_params=kwargs.get("backend_params", None),
                          use_trace=kwargs.get("use_trace", False),
                          inputs=kwargs.get("inputs", None),
                          outputs=kwargs.get("outputs", None),
                          environment_constants=kwargs.get(
                              "environment_constants", None),
                          register_buffers=kwargs.get("register_buffers", None))

        return node, name

    #########################################################
    # annotation APIs
    #########################################################
    def input_tensors(self, tensors, node_name: str):
        if not ExportManager._interpret_graph:
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)
        self._verify_no_active_function_tracing()

        if node_name in self.nodes.keys():
            traced_tensors_node = self.nodes[node_name]
        else:
            traced_tensors_node, node_name = self._setup_new_node(
                node_name, TracedTensorNode)
            self.nodes[node_name] = traced_tensors_node

        # TODO: this is still confusing. we need to make it more explicit.
        tensors_changed = False
        if not isinstance(tensors, dict):
            tensors_changed = True
            tensors = {'tensor': tensors}

        # reason this is convoluted is to mirror the scheme in output_tensors
        if tensors_changed:
            _get_logger().warning(f"Warning: no tensor name provided for input_tensors call in node {node_name}\n"
                                  "Assuming default tensor name")

        # if the node is not tracing, we validate the inputs only and return the raw tensors
        if not traced_tensors_node.is_tracing:
            for tensor_name, tensor in tensors.items():
                traced_tensors_node.validate_input_and_update_tags(
                    tensor_name, tensor_name, tensor)
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

            # TODO: need a scheme to update tags. in the future that scheme will be expanded to check
            # the inputs and outputs for type equivalence.

        # we need to handle input tensors more carefully than outputs because
        # we need to ensure the inputs are returned in the original structure
        traced_tensors = []
        for tensor_name, tensor in tensors.items():
            traced_tensor = traced_tensors_node.create_input(
                tensor, tensor_name)
            traced_tensors.append(traced_tensor)

        # if node is tracing we return the traced tensors
        return traced_tensors[0] if len(traced_tensors) == 1 else tuple(traced_tensors)

    def output_tensors(self, node_name: str, tensors, static_outputs=None, **kwargs):
        if not ExportManager._interpret_graph:
            return
        self._verify_no_active_function_tracing()

        if node_name in self.nodes.keys():
            traced_tensors_node = self.nodes[node_name]
        else:
            _get_logger().error(
                f"Error: output tensors called for node {node_name} but not registered to the ExportManager")
            raise Exception(
                "Error: exception detected in output_tensors declaration")
        
        ## process outputs
        tensors_changed = False
        if not isinstance(tensors, dict):
            tensors_changed = True
            tensors = {'tensor': tensors}

        flattened_tensors = flatten_io_structure(tensors, '')

        if not traced_tensors_node.is_tracing:
            # tag regardless of tracing status
            for tensor_name, tensor in flattened_tensors.items():
                traced_tensors_node.tag_data(tensor, tensor_name)
            return

        if tensors_changed:
            _get_logger().warning(f"Warning: no tensor name provided for output_tensors call in node {node_name}\n"
                                "Assuming default tensor name")

        types = set(type(tensor) for tensor in flattened_tensors.values())

        if not all([type is TracedTensor for type in types]):
            _get_logger().error(
                f"Error: detected the following types when expected all outputs to be TracedTensors: {types}\n"
                "**This could happen if you are not using TracedTensors in your computations.**\n"
                "Please verify if you are using the returned wrapped tensors from input_tensors() to "
                "correctly trace your computations.")
            raise Exception(
                "Error: exception detected in output_tensors declaration")

        context_names = set(
            [tensor.context for tensor in flattened_tensors.values()])
        if not len(context_names) == 1 and context_names.pop() != traced_tensors_node.name:
            _get_logger().error(
                f"Error: expected all context names to match the node name: {node_name}"
                f" but detected the following context names: {context_names}")
            raise Exception(
                "Error: exception detected in output_tensors declaration")
        
        ## process static outputs (constant tensors that should be returned but aren't derived from inputs)
        if static_outputs is not None:
            static_outputs_changed = False
            if not isinstance(static_outputs, dict):
                static_outputs_changed = True
                static_outputs = {'static_output': static_outputs}
            
            flattened_static_outputs = flatten_io_structure(static_outputs, '')
            
            if static_outputs_changed:
                _get_logger().warning(f"Warning: no tensor name provided for static_outputs in node {node_name}\n"
                                    "Assuming default tensor name")
            
            wrapped_static_outputs = traced_tensors_node.create_static_tensors(flattened_static_outputs)            
            
            # Merge with traced outputs
            flattened_tensors = {**flattened_tensors, **wrapped_static_outputs}

        traced_tensors_node.compile_trace(flattened_tensors,
                                        backend=kwargs.get("export_with", None),
                                        backend_params=kwargs.get("backend_params", {}))

    def block(self, node_name, **kwargs):
        """Create a context manager for tracing a block of code in the computational graph.

        This method initializes a context manager that traces a specific block of code when 
        used with a 'with' statement. It captures inputs, outputs, and execution details of 
        the code block to create a node in the LEAPP computational graph.

        Args:
            node_name (str): The unique name to identify this node in the computational graph.
            **kwargs: Additional parameters for node configuration. Supported options include:
                - export_with: Backend to use for exporting the model.
                - backend_params: Parameters for the export backend.
                - use_trace: Whether to use tracing for model compilation.
                - inputs: Input specifications for the node.
                - outputs: Output specifications for the node.
                - environment_constants: Constants to capture from the environment.
                - register_buffers: Buffers to register with the model.

        Returns:
            BlockTraceContext: A context manager for tracing the block.

        Example:
            ```python
            with export_manager.block("preprocessing_block"):
                # Code to be traced
                data = preprocess(raw_input)
                result = transform(data)
            ```

        Note:
            - Must be used with a 'with' statement to properly enter and exit tracing.
            - Graph interpretation must be enabled via start() before using this method.
            - The traced code block should not contain nested block() or method() annotations.
        """
        if not ExportManager._interpret_graph:
            return self  # no-op context manager

        if node_name in self.nodes.keys():
            new_node = False
            name = node_name
            node_context = self.nodes[node_name]
        else:
            new_node = True
            node_context, name = self._setup_new_node(
                node_name, BlockContextNode, **kwargs)
            self.nodes[name] = node_context
        export_manager = self

        class BlockTraceContext:
            """Context manager for tracing a block of code."""

            def __enter__(self):
                caller_frame = sys._getframe(1)
                # Convert frame to namespace immediately
                namespace = frame_to_namespace(caller_frame)

                if new_node:
                    # First entry - set the executed_lines boundaries
                    node_context.executed_lines.update({
                        'filename': caller_frame.f_code.co_filename,
                        'function_name': caller_frame.f_code.co_name,
                        'min_line': caller_frame.f_lineno,
                        'max_line': find_with_block_end(caller_frame.f_code.co_filename, caller_frame.f_lineno)
                    })
                    node_context.capture_inputs_from_namespace(namespace)
                    node_context.snapshot_buffer_values(namespace)
                    _get_logger().info(f"****Tracing started for {name}****")
                    node_context.compile_trace()
                else:
                    # Re-entry - validate boundaries and inputs match
                    node_context.validate_function_boundaries(caller_frame)
                    node_context.validate_inputs_from_namespace(namespace)
                # Acquire lock to prevent nested tracing and TracedTensor operations inside block
                TracingLock().acquire()

                return self

            def __exit__(self, exc_type, exc_value, traceback):
                TracingLock().release()
                # Convert frame to namespace for output capture
                output_namespace = frame_to_namespace(sys._getframe(1))
                if new_node:
                    _get_logger().info(
                        f"****Tracing stopped for {node_context.name}****\n\n")
                    node_context.capture_outputs_from_namespace(output_namespace)
                else:
                    node_context.validate_outputs_from_namespace(output_namespace)

        return BlockTraceContext()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

    def method(self, **params):
        """Create a decorator for tracing functions/methods in the computational graph.

        This method returns a decorator that wraps functions to trace their execution,
        capturing inputs, outputs, and execution details to create nodes in the LEAPP
        computational graph. The decorated function becomes a traceable node that can
        be connected with other nodes in the graph.

        Args:
            **params: Configuration parameters for the node. Supported options include:
                - node_name (str): Custom name for the node. If not provided, uses the
                  function's name.
                - export_with: Backend to use for exporting the model.
                - backend_params: Parameters for the export backend.
                - use_trace: Whether to use tracing for model compilation.
                - inputs: Input specifications for the node.
                - outputs: Output specifications for the node.
                - environment_constants: Constants to capture from the environment.
                - register_buffers: Buffers to register with the model.

        Returns:
            decorator: A decorator function that can be applied to functions/methods.

        Note:
            - Graph interpretation must be enabled via start() before decorated functions are called.
            - The decorator preserves the original function's metadata using functools.wraps.
            - Functions decorated with method() should not contain nested block() or method() annotations.
            - If graph interpretation is disabled, decorated functions execute normally without tracing.
        """
        def decorator(func):

            if "node_name" in params:
                name = params["node_name"]
            else:
                name = func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not ExportManager._interpret_graph:
                    return func(*args, **kwargs)

                # Check if this is a re-entry
                if name in self.nodes:
                    new_node = False
                    node_context = self.nodes[name]
                else:
                    new_node = True
                    node_context, _ = self._setup_new_node(
                        name, FunctionDecoratorNode, **params)
                    self.nodes[name] = node_context
                
                caller_namespace = frame_to_namespace(sys._getframe(1))

                if new_node:
                    _get_logger().info(f"****Tracing started for {name}****")
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    node_context.inspect_function_inputs(func, args, kwargs)
                    # Build merged namespace: caller frame + bound function arguments
                    # This allows capture and snapshot to look up 'self' and other function
                    # parameters that wouldn't be in the caller's frame.
                    input_namespace = {**caller_namespace, **bound_args.arguments}
                    # For bound methods, 'self' is not in bound_args (it's already bound)
                    # We need to explicitly add it from the method's __self__ attribute
                    if hasattr(func, '__self__'):
                        input_namespace['self'] = func.__self__
                    node_context.capture_inputs_from_namespace(input_namespace)
                    # Sets up function boundaries - required before tracing
                    node_context.compile_trace(func)
                    # Use the same namespace (with 'self' if bound method) for buffer/constant lookup
                    node_context.snapshot_buffer_values(input_namespace)

                    trace_fn = node_context.create_trace_function(
                        __file__.split('/')[-1], entry_hook=None)
                else:
                    node_context.validate_function_boundaries(func)
                    node_context.validate_function_inputs(func, args, kwargs)
                    # No entry_hook on re-entry - only need to capture output_namespace
                    trace_fn = node_context.create_trace_function(
                        __file__.split('/')[-1], entry_hook=None)

                # Start sys.settrace to capture namespaces (and run entry_hook on new_node)
                sys._getframe(1).f_trace = trace_fn
                
                # Acquire lock to prevent nested tracing and TracedTensor operations
                TracingLock().acquire()
                sys.settrace(trace_fn)

                try:
                    ##### run the actual function #########
                    result = func(*args, **kwargs)
                    #### run the actual function #########
                finally: 
                    sys.settrace(None)
                    TracingLock().release()

                if new_node:
                    _get_logger().info(
                        f"****Tracing stopped for {node_context.name}****\n\n")
                    node_context.inspect_function_outputs(func, result)
                    # capture outputs from the namespace for custom returns (declared via outputs=[...])
                    if node_context.output_namespace is not None:
                        node_context.capture_outputs_from_namespace(
                            node_context.output_namespace)
                else:
                    node_context.validate_function_outputs(func, result)
                    # validate outputs from namespace for declared outputs=[...]
                    if node_context.output_namespace is not None:
                        node_context.validate_outputs_from_namespace(
                            node_context.output_namespace)

                return result
            return wrapper
        return decorator

    #########################################################
    # export flow control
    #########################################################

    def mirror_leapp_tags(self, source, target):
        if not ExportManager._interpret_graph:
            return
        if TracingLock().is_active:
            raise Exception(
                "Error: detected calling mirror_leapp_tags while tracing a function/block. this function is only valid outside of nodes")
        try:
            if not verify_data_exact_match(source, target):
                _get_logger().error(
                    f"Error: source and target do not match: {source} != {target}")
                raise Exception("Error: source and target do not match")
            mirror_all_tensor_tags(source, target)
        except Exception as e:
            _get_logger().error(f"Unexpected error mirroring LEAPP tags: {e}")
            raise Exception(
                f"Error: unexpected error mirroring LEAPP tags: {e}")

    def get_io_descriptions(self):
        _get_logger().section(
            f"Compiling graph parameters for {len(self.nodes)} nodes")
        models = {"models": {}}
        nodes_to_describe = sorted(
            self.nodes.values(), key=lambda x: x.node_index)
        for node in nodes_to_describe:
            _get_logger().info(f"Compiling parameters for {node.name}")
            description = node.get_description()
            if 'parameters' in description and 'model_path' in description['parameters']:
                # Convert model path to be relative to YAML file location
                model_path = description['parameters']['model_path']
                if model_path:
                    description['parameters']['model_path'] = get_relative_path(
                        model_path, self.SAVE_PATH)
            models["models"][node.name] = description
        return models

    def compile_models(self):
        _get_logger().section(f"Discovered {len(self.nodes)} nodes")
        for node_context in self.nodes.values():
            _get_logger().section(f"Compiling {node_context.name}")
            node_context.compile_model()
            _get_logger().info("Success\n")

    def save_models(self):
        if self.SAVE_PATH is None:
            raise Exception(
                "Error: No save path provided, please provide a save path to export the graph")
        _get_logger().section(
            f"Saving {len(self.nodes)} models to {self.SAVE_PATH}")
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        for node_context in self.nodes.values():
            _get_logger().info(
                f"Saving {node_context.name}")
            node_context.save_model(self.SAVE_PATH)
            _get_logger().info("Success\n")

    #########################################################
    # graph compilation
    #########################################################
    def compile_graph(self, visualize=True, merge_nodes: MergeCfgEnum = MergeCfgEnum.NO_MERGE,
                      validate: bool = True, rtol: float = 1e-3, atol: float = 1e-5, strict=True):
        """Compile and save the computational graph from traced nodes.

        This method performs the complete pipeline of compiling traced nodes into exportable
        models, building graph connections, saving models to disk, and generating a YAML 
        description of the entire computational graph. It also optionally creates a 
        visualization of the graph structure.

        Args:
            visualize (bool, optional): If True, generates a visual representation of the 
                graph structure and saves it to the output directory. The visualization 
                will be created even if an error occurs during the process. 
                Defaults to True.
            merge_nodes (MergeCfgEnum, optional): Strategy for merging nodes in the graph.
                Options from MergeCfgEnum include:
                - NO_MERGE: Keep all nodes separate (default)
                - MERGE_ALL: Merge all possible nodes
                - MERGE_SEQUENTIAL: Merge only sequentially connected nodes (not available yet)
                Defaults to MergeCfgEnum.NO_MERGE.

        Returns:
            None

        Generated Artifacts:
            - Compiles all traced models using the configured backend
            - Saves compiled models to {SAVE_PATH}/ directory
            - Creates {GRAPH_NAME}.yaml file with complete graph description
            - Generates visualization files if visualize=True
            - Updates self.detected_nodes and self.detected_pipeline attributes
            - Prints graph statistics to the logger

        Note:
            - Must be called after tracing is complete and stop() has been called.
            - The YAML file contains model descriptions, pipeline connections, and system info.
            - Graph statistics include node count, dangling inputs/outputs, and edge counts.
            - Visualization errors are logged but don't stop the compilation process.
        """
        # compile models first before input name reconciliation
        self.compile_models()

        # builds the graph connections. this may change input and output names
        if not isinstance(merge_nodes, MergeCfgEnum):
            raise Exception(
                f"Error: merge_nodes must be an instance of MergeCfgEnum, got {type(merge_nodes)}")
        graph = LeappGraph(self.nodes)
        graph.merge_nodes(merge_nodes)
        pipeline = graph.get_full_pipeline_description()

        self.save_models()

        models = self.get_io_descriptions()

        if visualize:
            try:
                graph.visualize(self.SAVE_PATH, self.GRAPH_NAME)
            except Exception as e:
                _get_logger().error(f"Error visualizing graph: {e}")

        internal_connections, total_edges = graph.get_graph_statistics()

        # Print graph statistics
        _get_logger().section("Graph Statistics")
        _get_logger().info(f"- Computation nodes: {len(self.nodes)}")
        _get_logger().info(f"- Dangling inputs: {len(graph.graph_inputs)}")
        _get_logger().info(f"- Dangling outputs: {len(graph.graph_outputs)}")
        _get_logger().info(f"- Internal connections: {internal_connections}")
        _get_logger().info(f"- Total edges: {total_edges}")

        system_info = get_system_info()
        with open(os.path.join(self.SAVE_PATH, f"{self.GRAPH_NAME}.yaml"), "w") as f:
            yaml.dump(models, f, sort_keys=False)
            f.write("\n")  # Add a newline separator
            yaml.dump(pipeline, f)
            f.write("\n")
            yaml.dump(system_info, f)
            f.write("\n")

        # store the models and pipeline as part of the object
        # this will do a rewrite each time
        # this is **ONLY USED FOR TESTING**
        self.detected_nodes = models['models']
        self.detected_pipeline = pipeline['pipeline']

        # validate all the models in the compute graph
        if validate:
            return self.validate_all_models(rtol=rtol, atol=atol, strict=strict)

        return True

    def validate_all_models(self, rtol: float = 1e-3, atol: float = 1e-5, strict: bool = True):
        """Validate all exported models by comparing computed outputs against captured outputs.

        This method runs each compiled model with its captured input values and verifies
        that the outputs match the captured output values within the specified tolerances.

        Args:
            rtol (float): Relative tolerance for torch.allclose comparison. Default: 1e-3
            atol (float): Absolute tolerance for torch.allclose comparison. Default: 1e-5

        Returns:
            dict: A dictionary mapping node names to validation results:
                - True: All outputs matched within tolerance
                - False: At least one output did not match
                - Exception object: If model execution failed

        Raises:
            Exception: If called before compile_models() has been run.

        Example:
            >>> annotate.start(name="my_graph")
            >>> # ... run your code ...
            >>> annotate.stop()
            >>> annotate.compile_graph()
            >>> results = annotate.validate_all_models()
            >>> assert all(results.values()), "Some models failed validation"
        """

        if not self.nodes:
            _get_logger().error("No nodes to validate")
            return {}

        results = {}

        _get_logger().section(f"Validating {len(self.nodes)} models")

        for node_name, node in self.nodes.items():

            _get_logger().info(f"Validating {node_name}...")

            try:
                # Check that model has been compiled
                if node.compiled_model is None:
                    _get_logger().warning(f"Model {node_name} does not have a compiled model. "
                                          "Skipping validation.")
                    # model wasn't provided but we will skip validation
                    results[node_name] = True
                    continue

                # 1. Extract input values from TensorDescriptions
                # The compiled model expects flat tensor inputs in order
                input_values = [
                    tensor_desc.value for tensor_desc in node.inputs]

                # 3. Run the compiled model
                with torch.no_grad():
                    actual_outputs = node.compiled_model(*input_values)

                # 4. Normalize outputs to tuple for consistent handling
                if not isinstance(actual_outputs, tuple):
                    actual_outputs = (actual_outputs,)

                # 5. Extract expected output values
                expected_outputs = tuple(
                    tensor_desc.value for tensor_desc in node.outputs)

                # 6. Validate output count matches
                if len(actual_outputs) != len(expected_outputs):
                    _get_logger().error(
                        f"{node_name}: Output count mismatch - "
                        f"got {len(actual_outputs)}, expected {len(expected_outputs)}")
                    results[node_name] = True  # the model wasn't provided
                    continue

                # 7. Compare each output tensor
                all_match = True
                for idx, (actual, expected) in enumerate(zip(actual_outputs, expected_outputs)):
                    output_name = node.outputs[idx].name if idx < len(
                        node.outputs) else f"output_{idx}"

                    # Ensure tensors are on the same device for comparison
                    if actual.device != expected.device:
                        actual = actual.to(expected.device)

                    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
                        all_match = False
                        max_diff = (actual - expected).abs().max().item()
                        _get_logger().error(
                            f"{node_name}/{output_name}: Mismatch detected, "
                            f"max diff: {max_diff:.6e} (rtol={rtol}, atol={atol})")

                        _get_logger().info(
                            f"  Expected shape: {expected.shape}, dtype: {expected.dtype}")
                        _get_logger().info(
                            f"  Actual shape: {actual.shape}, dtype: {actual.dtype}")

                results[node_name] = all_match

                if all_match:
                    _get_logger().info(f"  ✓ {node_name} passed validation")

            except Exception as e:
                _get_logger().error(
                    f"{node_name}: Validation failed with exception: {e}")
                results[node_name] = e

        # Print summary
        _get_logger().section("Validation Summary")
        passed = sum(1 for v in results.values() if v is True)
        failed = sum(1 for v in results.values() if v is False)
        errors = sum(1 for v in results.values() if isinstance(v, Exception))

        _get_logger().info(f"  Passed: {passed}/{len(results)}")
        if failed > 0:
            _get_logger().warning(f"  Failed: {failed}/{len(results)}")
        if errors > 0:
            _get_logger().error(f"  Errors: {errors}/{len(results)}")

        if strict and (failed > 0 or errors > 0):
            failed_nodes = [name for name,
                            v in results.items() if v is not True]
            raise Exception(
                f"Model validation failed for {len(failed_nodes)} node(s): {failed_nodes}"
            )

        return results
