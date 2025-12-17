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
import yaml
import os
import torch

from leapp._logging import _get_logger
from leapp.leapp_graph.leapp_graph import LeappGraph
from leapp.leapp_graph.function_decorator_node import FunctionDecoratorNode
from leapp.leapp_graph.leapp_node import LeappNode
from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.traced_tensor import TracedTensor
from leapp.leapp_graph.block_context_node import BlockContextNode
from leapp.enums import MergeCfgEnum
from leapp.tracing_lock import TracingLock

from .utils import (CompactYamlList,
                    CompactYamlDict,
                    get_relative_path,
                    get_system_info,
                    verify_data_exact_match,
                    mirror_all_tensor_tags)


class ExportManager:
    _instance = None
    _initialized = False  # True after singleton __init__ completes
    _interpret_graph = False  # True between start() and stop() - enables graph interpretation

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
    # Tracing
    #########################################################

    def _start_tracing(self, frame, trace_function):
        """Start the trace function."""
        TracingLock().acquire()
        frame.f_trace = trace_function
        sys.settrace(trace_function)

    def _stop_tracing(self, node_name, node_context):
        if TracingLock().is_active:
            """Stop the trace function."""
            sys.settrace(None)

            try:
                if node_name in self.nodes.keys():
                    # we have seen this node before, we need to check if all values match.
                    original_node_data = {
                        k: v for k, v in self.nodes[node_name].executed_lines.items() if k != 'source_code'}
                    new_node_data = {
                        k: v for k, v in node_context.executed_lines.items() if k != 'source_code'}

                    if original_node_data != new_node_data:
                        _get_logger().error(
                            f"Error: {node_name} seen twice but detected lines do not match\n"
                            f"Original node data: {original_node_data}\n"
                            f"New node data: {new_node_data}")
                        raise Exception(
                            f"Error: {node_name} seen twice but detected lines do not match\n"
                            f"Original node data: {original_node_data}\n"
                            f"New node data: {new_node_data}")

                self.nodes[node_name] = node_context

            finally:
                # Always reset tracing state, even if an exception occurred
                TracingLock().release()

    #########################################################
    # annotation APIs
    #########################################################
    def input_tensors(self, tensors: dict[str, torch.Tensor], node_name: str):
        self._verify_no_active_function_tracing()
        if not ExportManager._interpret_graph:
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

        if node_name in self.nodes.keys():
            traced_tensors_node = self.nodes[node_name]
        else:
            traced_tensors_node, node_name = self._setup_new_node(
                node_name, TracedTensorNode)
            self.nodes[node_name] = traced_tensors_node

        # we run create tensors regardless to validate the inputs
        traced_tensors = []
        for tensor_name, tensor in tensors.items():
            traced_tensor = traced_tensors_node.create_input(
                tensor, tensor_name)
            traced_tensors.append(traced_tensor)

        # if the node is not tracing, we return the raw tensors
        if not traced_tensors_node.is_tracing:
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

        #if node is tracing we return the traced tensors
        return traced_tensors[0] if len(traced_tensors) == 1 else tuple(traced_tensors)

    def output_tensors(self, tensors: dict[str, TracedTensor], **kwargs):
        self._verify_no_active_function_tracing()
        if not ExportManager._interpret_graph:
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

        # validate if valid input is given to this function
        # if none of the tensors are traced, we assume tracing is not active and return the raw tensors
        if all(not isinstance(tensor, TracedTensor) for tensor in tensors.values()):
            return tuple(tensors.values())

        # if there is a mix of traced and raw tensors, we error
        context_names = []
        for name, tensor in tensors.items():
            if not isinstance(tensor, TracedTensor):
                _get_logger().error(
                    f"Error: tensor {name} is not a TracedTensor")
                raise Exception(
                    "Error: exeption detected in output_tensors declaration")
            context_names.append(tensor.context)
        if not all(name == context_names[0] for name in context_names):
            _get_logger().error(
                f"Error: all tensors must have the same context name, got {set(context_names)}")
            raise Exception(
                "Error: exeption detected in output_tensors declaration")
        node_name = context_names[0]
        if node_name not in self.nodes.keys():
            _get_logger().error(
                f"Error: output tensors declared for node {node_name} but not registered to the ExportManager")
            raise Exception(
                "Error: exeption detected in output_tensors declaration")
        node_context = self.nodes[node_name]
        if not node_context.is_tracing:
            _get_logger().error(f"Error: output tensors called on a node {node_name} that is not currently tracing."
                                " Please ensure one call only to output_tensors is made per node.")
            raise Exception(
                "Error: exeption detected in output_tensors declaration")

        node_context.compile_trace(tensors,
                                   backend=kwargs.get("export_with", None),
                                   backend_params=kwargs.get("backend_params", {}))

        raw_tensors = list(tensor.tensor for tensor in tensors.values())
        return raw_tensors[0] if len(raw_tensors) == 1 else tuple(raw_tensors)

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
                - enable_fp16: Enable FP16 precision mode.
                - enable_cuda_graphs: Enable CUDA graphs optimization.

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

        node_context, name = self._setup_new_node(
            node_name, BlockContextNode, **kwargs)
        export_manager = self
        skip_file = __file__.split('/')[-1]

        class BlockTraceContext:
            """Context manager for tracing a block of code."""

            def __enter__(self):
                caller_frame = sys._getframe(1)

                node_context.capture_inputs_from_frame(caller_frame)

                _get_logger().info(f"****Tracing started for {name}****")

                # Set up local tracing for the caller frame
                if hasattr(caller_frame, 'f_trace_lines'):
                    caller_frame.f_trace_lines = True
                node_context.executed_lines['filename'] = caller_frame.f_code.co_filename
                node_context.executed_lines['function_name'] = caller_frame.f_code.co_name
                node_context.executed_lines['min_line'] = caller_frame.f_lineno
                node_context.executed_lines['max_line'] = caller_frame.f_lineno

                trace_fn = node_context.create_trace_function(skip_file)
                export_manager._start_tracing(caller_frame, trace_fn)
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                export_manager._stop_tracing(name, node_context)

                node_context.compile_trace()
                node_context.capture_outputs_from_frame(sys._getframe(1))
                _get_logger().info(
                    f"****Tracing stopped for {node_context.name}****\n\n")

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
                - enable_fp16: Enable FP16 precision mode.
                - enable_cuda_graphs: Enable CUDA graphs optimization.

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

                node_context, node_name = self._setup_new_node(
                    name, FunctionDecoratorNode, **params)

                _get_logger().info(f"****Tracing started for {name}****")
                node_context.inspect_function_inputs(
                    func, args, kwargs)
                # tracing requires max and min lines to be configured already so this needs to be run before tracing
                node_context.compile_trace(func)

                # this tracing step captures the input and output frames
                trace_fn = node_context.create_trace_function(
                    __file__.split('/')[-1])
                self._start_tracing(sys._getframe(1), trace_fn)

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    raise e
                finally:
                    self._stop_tracing(
                        node_name, node_context)
                    if name not in self.nodes.keys():
                        _get_logger().error(
                            f"Error: Tracing stopped for {name} but node not found in nodes dictionary")
                        raise Exception("Error: in annotating method")
                    node_context = self.nodes[name]
                    _get_logger().info(
                        f"****Tracing stopped for {node_context.name}****\n\n")

                node_context.inspect_function_outputs(func, result)
                # capture outputs from the frame for custom returns
                if node_context.output_frame is not None:
                    node_context.capture_outputs_from_frame(
                        node_context.output_frame)
                return result
            return wrapper
        return decorator

    #########################################################
    # export flow control
    #########################################################

    def mirror_leapp_tags(self, source, target):
        if not ExportManager._interpret_graph:
            return
        try:
            if not verify_data_exact_match(source, target):
                _get_logger().error(
                    f"Error: source and target do not match: {source} != {target}")
                raise Exception("Error: source and target do not match")
            mirror_all_tensor_tags(source, target)
        except Exception as e:
            _get_logger().error(f"Unexpected error mirroring LEAPP tags: {e}")
            raise Exception(f"Error: unexpected error mirroring LEAPP tags: {e}")

    def get_io_descriptions(self):
        _get_logger().section(
            f"Compiling graph parameters for {len(self.nodes)} nodes")
        models = {"models": {}}
        for node in self.nodes.values():
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
    def compile_graph(self, visualize=True, merge_nodes: MergeCfgEnum = MergeCfgEnum.NO_MERGE):
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
            yaml.dump(models, f)
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
