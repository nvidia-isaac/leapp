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
import functools
import torch
from leapp.utils.tensor_description import (
    describe_io,
    flatten_io_structure,
    TensorSemantics,
)
from leapp.leapp_graph.datatypes import is_traced_type
from leapp.backends.export_backend import NoneExportBackend
from leapp.leapp_graph.custom_operator_registry import prepare_and_validate
from leapp.leapp_graph.custom_operator_registry.warp_operator.bundle import (
    iter_warp_segments_from_graph,
)
from leapp.utils.logging import _get_logger


class LeappNode():
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '__init__' in cls.__dict__:
            original_init = cls.__init__
            if getattr(original_init, '_leapp_wrapped', False):
                return

            @functools.wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                # Only log if this is the actual class being instantiated
                # (not a parent's __init__ being called via super())
                if type(self) is cls:
                    _get_logger().info(
                        f"Node context initialized: {self.name}")
            wrapped_init._leapp_wrapped = True
            cls.__init__ = wrapped_init

        # Wrap compile_trace to set _model_captured = True after execution
        if 'compile_trace' in cls.__dict__:
            original_compile_trace = cls.compile_trace

            @functools.wraps(original_compile_trace)
            def wrapped_compile_trace(self, *args, **kwargs):
                result = original_compile_trace(self, *args, **kwargs)
                self._model_captured = True
                return result
            cls.compile_trace = wrapped_compile_trace

    UNSET_NODE_INDEX = -1

    def __init__(self, name):
        self.name = name
        self._node_index = self.UNSET_NODE_INDEX
        
        # Attributes expected by export backends (subclasses may override)
        self.register_buffers = set()
        self.environment_constants = set()

        # model settings
        self._model_captured = False
        self.model_path = None
        self.md5sum = None
        self.sha256sum = None
        self.export_backend = NoneExportBackend(self, {})
        self.backend = None

        # i/o settings
        self.inputs = []
        self.outputs = []
        self.input_formats = []
        self.output_formats = []
        # trimmed inputs are inputs that are not used in the computation or directly returned as output
        self.trimmed_inputs = set()
        # caller identities track which call sites created this node's inputs
        self._caller_identities = set()

        # i/o caching for multi-example validation
        self._max_cached_io = 0
        self._cache_write_idx = 0

        # storage for the fx graph or compiled module
        self.m = None

    @property
    def node_index(self):
        return self._node_index

    @node_index.setter
    def node_index(self, value):
        if self._node_index != self.UNSET_NODE_INDEX:
            _get_logger().fatal(
                f"Node index for '{self.name}' is already set to {self._node_index} "
                "and cannot be reassigned.",
                error_type=Exception)
        self._node_index = value

    @property
    def captured(self):
        # this is defaulted to False. the compile_trace method should set this to true
        return self._model_captured

    @property
    def has_pending_warp_segments(self) -> bool:
        return False

    @property
    def exports_model(self) -> bool:
        """Whether this node compiles and saves a model artifact.

        ``False`` covers ``export_with=None`` as well as the dry-run and
        ``non_traced`` cases, which are both routed to ``export_with=None``.
        Work that only feeds export can be skipped when this is ``False``.
        """
        return self.backend not in (None, "None")
    
    @property
    def compiled_model(self):
        if self.export_backend is None:
            return None
        if self.export_backend.compiled_model is None:
            return None
        return self.export_backend.compiled_model
    
    @property
    def compiled_module(self):
        if self.export_backend is None:
            return None
        if self.export_backend.compiled_module is None:
            return None
        return self.export_backend.compiled_module
    
    def delete_compiled_model(self):
        if self.export_backend is None:
            return
        self.export_backend.compiled_model = None
        self.export_backend.compiled_module = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_description(self):
        # dynamically generate i/o descriptions depending on need
        # Directly use the TensorDescription objects in self.inputs
        input_descriptions = [input.dict() for input in self.inputs]

        # Directly use the TensorDescription objects in self.outputs
        output_descriptions = [output.dict() for output in self.outputs]

        parameters = {
            'model_path': self.model_path,
            'md5sum': self.md5sum,
            'sha256sum': self.sha256sum,
            'backend': self.get_backend(),
            'warp_segments': len(iter_warp_segments_from_graph(self.graph))
            if hasattr(self, 'graph') else 0,
        }

        backend_metadata = self.export_backend.get_backend_metadata()
        if backend_metadata:
            parameters.update(backend_metadata)

        description = {}
        description['inputs'] = input_descriptions
        description['outputs'] = output_descriptions
        description['parameters'] = parameters

        return description

    def setup_backend(self, backend, backend_params):
        self._create_backend(backend, backend_params)

    def _create_backend(self, backend, backend_params):
        available_backends = [
            "jit-script", "jit-trace", "onnx-dynamo", "onnx-torchscript",
            "exported-program",
        ]
        if backend is None:
            self.backend = "None"
            self.export_backend = NoneExportBackend(
                self, backend_params)
        elif backend == "jit-script" or backend == "jit": # default jit export method
            self.backend = "jit-script"
            from leapp.backends.torch_export_backend import TorchScriptExportBackend
            self.export_backend = TorchScriptExportBackend(
                self, backend_params)
        elif backend == "jit-trace":
            self.backend = "jit-trace"
            from leapp.backends.torch_export_backend import TorchTraceExportBackend
            self.export_backend = TorchTraceExportBackend(
                self, backend_params)
        elif backend == "onnx-dynamo" or backend == "onnx": #default onnx export method
            self.backend = "onnx-dynamo"
            from leapp.backends.onnx_export_backend import ONNXDynamoExportBackend
            self.export_backend = ONNXDynamoExportBackend(
                self, backend_params)
        elif backend == "onnx-torchscript":
            self.backend = "onnx-torchscript"
            from leapp.backends.onnx_export_backend import ONNXTorchScriptExportBackend
            self.export_backend = ONNXTorchScriptExportBackend(
                self, backend_params)
        elif backend == "exported-program" or backend == "pt2":
            self.backend = "exported-program"
            from leapp.backends.exported_program_backend import ExportedProgramExportBackend
            self.export_backend = ExportedProgramExportBackend(
                self, backend_params)
        elif backend == "cpp":
            _get_logger().fatal("C++ backend not implemented", error_type=Exception)
        elif backend == "py":
            _get_logger().fatal("Python backend not implemented", error_type=Exception)
        else:
            _get_logger().fatal(
                f"{self.name} Unexpected backend: {backend}, \n"
                f"please use one of the following: {available_backends}",
                error_type=Exception)

    def save_model(self, save_path: str):
        self.model_path, self.md5sum, self.sha256sum = self.export_backend.save(save_path)

    def compile_model(self):
        try:
            if self.exports_model:
                prepare_and_validate(self.m, self.backend)
            self.export_backend.compile(self.m)
        except Exception as e:
            _get_logger().fatal(
                f"Error compiling model {self.name}: {e}",
                error_type=type(e),
                cause=e)

    def get_backend(self):
        return self.export_backend.get_backend_model_type()

    @property
    def is_tracing(self) -> bool:
        """Whether this node is currently recording operations.
        """
        return False

    def publish_output_port(self, value, port: str):
        """Record that ``value`` is this node's output named ``port``.

        A traced value already carries the node that owns it, so the port name
        completes the connection identity that consuming nodes read back. Raw
        values cannot carry state and are connected through this node's own
        descriptions instead.
        """
        if not is_traced_type(value):
            return value
        existing = value.output_port
        if existing is not None and existing != port:
            # One value can legitimately be several outputs, most commonly a
            # state update that is also published for downstream use. Each port
            # delivers the same data, so any of them is a correct edge and the
            # first stays as this value's identity. Both are still exported,
            # and state feedback is wired from the descriptions regardless.
            _get_logger().debug(
                f"Node '{self.name}' publishes one value as both output "
                f"'{existing}' and output '{port}'; consumers of it will "
                f"connect through '{existing}'.")
            return value
        value.output_port = port
        return value

    def publish_output_ports(self, value, name: str, descriptions):
        """Assign final output-port names across a possibly nested output value.

        ``descriptions`` come from ``describe_io``, which flattens the value in
        the same order as ``flatten_io_structure``, so the two line up leaf by
        leaf even after semantics rename a description.
        """
        leaves = list(flatten_io_structure(value, name).values())
        if len(leaves) != len(descriptions):
            _get_logger().error(
                f"Error: output '{name}' of node '{self.name}' flattened to "
                f"{len(leaves)} values but {len(descriptions)} descriptions; "
                "skipping output-port registration for it.")
            return
        for leaf, description in zip(leaves, descriptions):
            self.publish_output_port(leaf, description.port)

    @staticmethod
    def _validate_and_add_to_list(descriptions, current_io_list, node_name):
        existing_names = set([io.name_str for io in current_io_list])
        for description in descriptions:
            if description.name_str in existing_names:
                _get_logger().fatal(
                    f"Duplicate i/o name '{description.name_str}' in node '{node_name}'. \n"
                    f"Currently existing names: {existing_names}\n"
                    f"Each input/output in the same node must have a unique name.",
                    error_type=Exception)
            existing_names.add(description.name_str)
            current_io_list.append(description)

    @staticmethod
    def _apply_semantics_to_descriptions(io_descriptions, semantics):
        if semantics is None:
            return
        if len(io_descriptions) == 1:
            io_descriptions[0].init_semantics(semantics)
            return

        semantic_fields = semantics.to_dict()
        for desc in io_descriptions:
            desc.init_semantics(TensorSemantics(
                name=desc.name,
                ref=desc.value,
                **semantic_fields,
            ))

    def add_output(self, output_name, raw_output_name, output_value, semantics=None):
        io_descriptions, output_format = describe_io(
            output_name, raw_output_name, output_value)
        self._apply_semantics_to_descriptions(io_descriptions, semantics)
        for desc in io_descriptions:
            desc.cached_values = [torch.zeros_like(desc.value) for _ in range(self._max_cached_io)]
        self._validate_and_add_to_list(
            io_descriptions, self.outputs, self.name)
        for desc in io_descriptions:
            # An output is its own source. The name was just proven unique on
            # this node, which is what makes it usable as the port identity.
            desc.source_node = self
            desc.port = desc.name_str
        # used to rebuild the nested i/o
        self.output_formats.append(output_format)
        return io_descriptions

    def add_input(self, input_name, raw_input_name, input_value, semantics=None):
        io_descriptions, input_format = describe_io(
            input_name, raw_input_name, input_value)
        self._apply_semantics_to_descriptions(io_descriptions, semantics)
        for desc in io_descriptions:
            desc.cached_values = [torch.zeros_like(desc.value) for _ in range(self._max_cached_io)]
        self._validate_and_add_to_list(io_descriptions, self.inputs, self.name)
        # used to rebuild the nested i/o
        self.input_formats.append(input_format)
        return io_descriptions

    def state_feedback_pairs(self):
        """Input/output description pairs this node feeds back into itself.

        State tensors are declared as pairs rather than discovered from the
        values crossing a node boundary, so their edges are built from the
        declaration and never depend on what a fed-back value carries.
        """
        return ()

    @staticmethod
    def get_io_description_by_name(name, io_list):
        for io_description in io_list:
            if name in io_description.aliases:
                return io_description
        return None

    def validate_io_and_update_sources(self, io_name, raw_io_name, io_value, current_io_list):
        '''
        this is used for rerunning the the tracing. each time we run it we
        we want to validate that the inputs are consistent with the previous run
        and also resolve sources that only appear on a later pass, which is how
        feedback edges are discovered.
        '''
        io_descriptions, _ = describe_io(io_name, raw_io_name, io_value)
        for io_description in io_descriptions:
            if io_description.name_str in self.trimmed_inputs:
                # we skip validation for inputs that are not used in the model
                continue
            existing_io_description = LeappNode.get_io_description_by_name(
                io_description.name_str, current_io_list)
            if existing_io_description is None:
                _get_logger().fatal(
                    f"Error: Reentering {self.name} with new i/o {io_description.name_str} but failed to find it in the current i/o list.\n"
                    f"available i/o names: {[io.name_str for io in current_io_list]}",
                    error_type=Exception)
            elif current_io_list is self.inputs and not existing_io_description.has_source:
                # A previously dangling input may acquire its source on a later
                # pass; this is how feedback edges become visible. Only an input
                # can gain a source this way, because an output is its own
                # source from the moment it was registered.
                existing_io_description.source_node = io_description.source_node
                existing_io_description.port = io_description.port

            existing_io_description_dict = existing_io_description.dict()
            current_io_description_dict = io_description.dict()
            current_io_description_dict["name"] = existing_io_description.name

            common_keys = existing_io_description_dict.keys() & current_io_description_dict.keys()
            if not all(existing_io_description_dict[key] == current_io_description_dict[key] for key in common_keys):
                _get_logger().fatal(
                    f"Error: Reentering {self.name} with new i/o {io_description.name_str} \n"
                    f"but the description has changed from {existing_io_description_dict} to {current_io_description_dict}.\n"
                    f"This can happen if some dynamic behavior is not captured by the annotations",
                    error_type=Exception)

            if self._cache_write_idx < self._max_cached_io:
                existing_io_description.cached_values[self._cache_write_idx] = io_description.value

    def increment_cache_idx(self):
        if self._cache_write_idx < self._max_cached_io:
            self._cache_write_idx += 1

    def _build_validation_examples(self):
        examples = [(0,
            [td.value for td in self.inputs],
            tuple(td.value for td in self.outputs))]

        for cache_idx in range(self._cache_write_idx):
            cached_inputs = [td.cached_values[cache_idx] for td in self.inputs]
            cached_outputs = tuple(td.cached_values[cache_idx] for td in self.outputs)
            examples.append((cache_idx + 1, cached_inputs, cached_outputs))
        return examples

    def _reentry_validation_hint(self, sample_passed: dict) -> str | None:
        """Return a hint when only the initial trace passes and re-entry samples fail."""
        later_indices = [idx for idx in sample_passed if idx > 0]
        if not later_indices:
            return None
        if not sample_passed.get(0):
            return None
        if not any(not sample_passed[idx] for idx in later_indices):
            return None
        return (
            "Sample 0 (initial trace) passed validation, but one or more re-entry "
            "samples failed. This pattern often means a value that changes across "
            "iterations was inlined as a constant during export. Consider declaring "
            "it with annotate.input_tensors() for this node."
        )

    def validate_compiled_model(self, rtol: float = 1e-3, atol: float = 1e-5) -> tuple[bool, str | None]:
        if self.compiled_model is None:
            _get_logger().warning(
                f"Model {self.name} does not have a compiled model. Skipping validation.")
            return True, None

        examples = self._build_validation_examples()
        all_match = True
        sample_passed = {}
        for sample_idx, input_values, source_outputs in examples:
            sample_passed[sample_idx] = True
            with torch.no_grad():
                exported_outputs = self.compiled_model(*input_values)

            if not isinstance(exported_outputs, tuple):
                exported_outputs = (exported_outputs,)

            if len(exported_outputs) != len(source_outputs):
                _get_logger().error(
                    f"{self.name} sample {sample_idx}: Output count mismatch - "
                    f"got {len(exported_outputs)}, expected {len(source_outputs)}")
                all_match = False
                sample_passed[sample_idx] = False
                continue

            for idx, (exported, source) in enumerate(zip(exported_outputs, source_outputs)):
                output_name = self.outputs[idx].name if idx < len(
                    self.outputs) else f"output_{idx}"

                if exported.device != source.device:
                    exported = exported.to(source.device)

                exported_nan = torch.isnan(exported).sum().item()
                exported_inf = torch.isinf(exported).sum().item()
                source_nan = torch.isnan(source).sum().item()
                source_inf = torch.isinf(source).sum().item()

                if exported_nan > 0 or exported_inf > 0 or source_nan > 0 or source_inf > 0:
                    all_match = False
                    sample_passed[sample_idx] = False
                    num_elements = exported.numel()
                    _get_logger().error(
                        f"{self.name}/{output_name} sample {sample_idx}: NaN/Inf detected!")
                    if exported_nan > 0:
                        _get_logger().error(
                            f"  Exported has {exported_nan}/{num_elements} NaN values ({100*exported_nan/num_elements:.3f}%)")
                    if exported_inf > 0:
                        _get_logger().error(
                            f"  Exported has {exported_inf}/{num_elements} Inf values ({100*exported_inf/num_elements:.3f}%)")
                    if source_nan > 0:
                        _get_logger().warning(
                            f"  Source has {source_nan}/{num_elements} NaN values ({100*source_nan/num_elements:.3f}%)")
                    if source_inf > 0:
                        _get_logger().warning(
                            f"  Source has {source_inf}/{num_elements} Inf values ({100*source_inf/num_elements:.3f}%)")
                    continue

                if not torch.allclose(exported, source, rtol=rtol, atol=atol):
                    all_match = False
                    sample_passed[sample_idx] = False
                    diff = (exported - source).abs()
                    diff_flat = diff.flatten().float()

                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()

                    percentiles = torch.tensor([0.50, 0.75, 0.90, 0.99, 0.995], device=diff_flat.device)
                    pct_values = torch.quantile(diff_flat, percentiles)
                    p50, p75, p90, p99, p995 = pct_values.tolist()

                    source_min, source_max = source.min().item(), source.max().item()
                    exported_min, exported_max = exported.min().item(), exported.max().item()

                    log_path = _get_logger().path
                    _get_logger().error(
                        f"{self.name}/{output_name} sample {sample_idx}: Mismatch detected (rtol={rtol}, atol={atol}). Please check {log_path} for more details.")
                    _get_logger().info(
                        f"  Source shape: {source.shape}, dtype: {source.dtype}")
                    _get_logger().info(
                        f"  Exported shape: {exported.shape}, dtype: {exported.dtype}")
                    _get_logger().info(
                        f"  Source range:   [{source_min:.6e}, {source_max:.6e}]")
                    _get_logger().info(
                        f"  Exported range: [{exported_min:.6e}, {exported_max:.6e}]")
                    _get_logger().info(
                        f"  Diff stats: max={max_diff:.6e}, mean={mean_diff:.6e}")
                    _get_logger().info(
                        f"  Diff percentiles: p50={p50:.6e}, p75={p75:.6e}, p90={p90:.6e}, p99={p99:.6e}, p995={p995:.6e}")

        error_hint = self._reentry_validation_hint(sample_passed)
        if error_hint is not None:
            _get_logger().warning(f"  {self.name}: {error_hint}")

        if all_match:
            num_examples = len(examples)
            _get_logger().info(
                f"  [PASS] {self.name} passed validation ({num_examples} example{'s' if num_examples > 1 else ''})")
        return all_match, error_hint

    def validate_input_and_update_sources(self, input_name, raw_input_name, input_value):
        self.validate_io_and_update_sources(
            input_name, raw_input_name, input_value, self.inputs)

    def validate_output_and_update_sources(self, output_name, raw_output_name, output_value):
        self.validate_io_and_update_sources(
            output_name, raw_output_name, output_value, self.outputs)

    def reentry_validate_inputs(self, tensors: dict):
        """Validate input tensors on re-entry (node already compiled)."""
        for tensor_name, tensor in tensors.items():
            self.validate_input_and_update_sources(
                tensor_name, tensor_name, tensor)

    def reentry_validate_outputs(self, tensors: dict,
                                 static_tensors: dict | None = None):
        """Validate output tensors on re-entry, then advance the cache index.

        Ports are not re-published here. The descriptions registered on the
        first pass already own this node's side of every edge, so a later pass
        only has to prove the data still looks the same.

        The tensors arrive already flattened, so each one is a single leaf.
        """
        all_tensors = {**tensors, **(static_tensors or {})}
        for tensor_name, tensor in all_tensors.items():
            self.validate_output_and_update_sources(
                tensor_name, tensor_name, tensor)

        self.increment_cache_idx()

    def reentry_validate_state_update(self, tensors: dict):
        """Validate state tensor updates on re-entry."""
        for tensor_name, tensor in tensors.items():
            self.validate_output_and_update_sources(
                tensor_name, tensor_name, tensor)

    def change_input_name(self, old_name, new_name):
        _get_logger().debug(
            f"changing input name from {old_name} to {new_name} for model {self.name}")
        if old_name == new_name:
            return
        current_input_names = [input.name_str for input in self.inputs]
        if new_name in current_input_names:
            _get_logger().fatal(
                f"Error requesting input name change for {self.name}/{old_name}:"
                f" {new_name} is already in use",
                error_type=Exception,
            )
        for input in self.inputs:
            if input.name_str == old_name:
                input.change_name(new_name)

    def change_output_name(self, old_name, new_name):
        _get_logger().debug(
            f"changing output name from {old_name} to {new_name} for model {self.name}")
        if old_name == new_name:
            return
        current_output_names = [output.name_str for output in self.outputs]
        if new_name in current_output_names:
            _get_logger().fatal(
                f"Error requesting output name change for {self.name}:"
                f" {new_name} is already in use",
                error_type=Exception,
            )
        for output in self.outputs:
            if output.name_str == old_name:
                output.change_name(new_name)

    def compile_trace(self, *args):
        raise NotImplementedError(
            f"compile_trace not implemented for {self.__class__.__name__}")
