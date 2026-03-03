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
import collections
import functools
import torch
from leapp.utils.tensor_description import describe_io
from leapp.utils.utils import tag_tensor
from leapp.leapp_graph.datatypes import is_tracable_tensor_type
from leapp.backends.export_backend import NoneExportBackend
from leapp._logging import _get_logger


class LeappNode():
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only wrap if this class defines its own __init__
        if '__init__' in cls.__dict__:
            original_init = cls.__init__

            @functools.wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                # Only log if this is the actual class being instantiated
                # (not a parent's __init__ being called via super())
                if type(self) is cls:
                    _get_logger().info(
                        f"Node context initialized: {self.name}")
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

    def __init__(self, name, node_index):
        self.name = name
        self.node_index = node_index

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
    def captured(self):
        # this is defaulted to False. the compile_trace method should set this to true
        return self._model_captured
    
    @property
    def compiled_model(self):
        if self.export_backend is None:
            return None
        if self.export_backend.compiled_model is None:
            return None
        return self.export_backend.compiled_model
        # assert self.export_backend is not None, f"Error: {self.name} has no export backend, please setup the backend first"
        # assert self.export_backend.compiled_model is not None, f"Error: {self.name} has no compiled model, please compile the model first"
        # return self.export_backend.compiled_model
    
    @property
    def compiled_module(self):
        if self.export_backend is None:
            return None
        if self.export_backend.compiled_module is None:
            return None
        return self.export_backend.compiled_module
        # assert self.export_backend is not None, f"Error: {self.name} has no export backend, please setup the backend first"
        # assert self.export_backend.compiled_module is not None, f"Error: {self.name} has no compiled module, please compile the model first"
        # return self.export_backend.compiled_module
    
    def delete_compiled_model(self):
        if self.export_backend is None:
            return
        if self.export_backend.compiled_model is not None:
            del self.export_backend.compiled_model
        if self.export_backend.compiled_module is not None:
            del self.export_backend.compiled_module

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
        available_backends = ["jit-script", "jit-trace", "onnx-dynamo", "onnx-torchscript"]
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
        elif backend == "cpp":
            raise Exception("C++ backend not implemented")
        elif backend == "py":
            raise Exception("Python backend not implemented")
        else:
            raise Exception(
                f"{self.name} Unexpected backend: {backend}, \n"
                f"please use one of the following: {available_backends}")

    def save_model(self, save_path: str):
        self.model_path, self.md5sum, self.sha256sum = self.export_backend.save(save_path)

    def compile_model(self):
        try:
            self.export_backend.compile(self.m)
        except Exception as e:
            _get_logger().error(f"Error compiling model {self.name}: {e}")
            raise e

    def get_backend(self):
        return self.export_backend.get_backed_model_type()

    def tag_data(self, tensor, tag):
        # the tag is the name of the tensor, with the node name prepended
        tag = self.name + '/' + tag + '/'

        if is_tracable_tensor_type(tensor):
            # Tag the tensor directly (works for both TracedTensor and regular tensors)
            # For TracedTensor, we tag it directly so the tag is preserved through operations
            tag_tensor(tensor, tag)
        elif isinstance(tensor, collections.abc.Mapping):
            for key, value in tensor.items():
                self.tag_data(value, tag + "[" + key + "]")
        elif isinstance(tensor, collections.abc.Iterable) and not isinstance(tensor, (str, bytes)) and not hasattr(tensor, '__array__'):
            # This catches lists, tuples, sets, etc. but excludes strings, bytes, and numpy arrays
            for idx, item in enumerate(tensor):
                self.tag_data(item, tag + "[" + str(idx) + "]")
        else:
            _get_logger().warning(
                f"\033[93mWarning: Untaggable datatype in i/o: {type(tensor)}\033[0m")

    @staticmethod
    def _validate_and_add_to_list(descriptions, current_io_list, node_name):
        existing_names = set([io.name_str for io in current_io_list])
        for description in descriptions:
            if description.name_str in existing_names:
                _get_logger().error(
                    f"Duplicate i/o name '{description.name_str}' in node '{node_name}'. \n"
                    f"Currently existing names: {existing_names}\n"
                    f"Each input/output in the same node must have a unique name."
                )
                raise Exception(
                    f"Duplicate name '{description.name_str}' in node '{node_name}'. "
                    f"Each input/output must have a unique name."
                )
            existing_names.add(description.name_str)
            current_io_list.append(description)

    def add_output(self, outout_name, raw_output_name, output_value):
        io_descriptions, output_format = describe_io(
            outout_name, raw_output_name, output_value)
        for desc in io_descriptions:
            desc.cached_values = [torch.zeros_like(desc.value) for _ in range(self._max_cached_io)]
        self._validate_and_add_to_list(
            io_descriptions, self.outputs, self.name)
        # used to rebuild the nested i/o
        self.output_formats.append(output_format)

    def add_input(self, input_name, raw_input_name, input_value):
        io_descriptions, input_format = describe_io(
            input_name, raw_input_name, input_value)
        for desc in io_descriptions:
            desc.cached_values = [torch.zeros_like(desc.value) for _ in range(self._max_cached_io)]
        self._validate_and_add_to_list(io_descriptions, self.inputs, self.name)
        # used to rebuild the nested i/o
        self.input_formats.append(input_format)

    @staticmethod
    def get_io_description_by_name(name, io_list):
        for io_description in io_list:
            if io_description.name_str == name:
                return io_description
        return None

    def validate_io_and_update_tags(self, io_name, raw_io_name, io_value, current_io_list):
        '''
        this is used for rerunning the the tracing. each time we run it we 
        we want to validate that the inputs are consistent with the previous run
        and also update tags for feedback detection.   
        '''
        io_descriptions, _ = describe_io(io_name, raw_io_name, io_value)
        for io_description in io_descriptions:
            if io_description.name_str in self.trimmed_inputs:
                # we skip validation for inputs that are not used in the model
                continue
            existing_io_description = LeappNode.get_io_description_by_name(
                io_description.name_str, current_io_list)
            if existing_io_description is None:
                _get_logger().error(
                    f"Error: Reentering {self.name} with new i/o {io_description.name_str} but failed to find it in the current i/o list.\n"
                    f"available i/o names: {[io.name_str for io in current_io_list]}"
                )
                raise Exception("Validation error when reentering node")
            elif existing_io_description.tag is None:
                # THIS STEP UPDATES THE TAG FOR FEEDBACK DETECTION
                existing_io_description.tag = io_description.tag
            elif io_description.tag is not None and existing_io_description.tag != io_description.tag:
                _get_logger().error(
                    f"Error: Reentering {self.name} with new i/o {io_description.name_str} \n"
                    f"but the tag has changed from {existing_io_description.tag} to {io_description.tag}.\n"
                    f"This can happen if some dynamic behavior is not captured by the annotations"
                )
                raise Exception("Validation error when reentering node")

            existing_io_description_dict = existing_io_description.dict()
            current_io_description_dict = io_description.dict()

            if not all([existing_io_description_dict[key] == current_io_description_dict[key] for key in existing_io_description_dict.keys()]):
                _get_logger().error(
                    f"Error: Reentering {self.name} with new i/o {io_description.name_str} \n"
                    f"but the description has changed from {existing_io_description_dict} to {current_io_description_dict}.\n"
                    f"This can happen if some dynamic behavior is not captured by the annotations"
                )
                raise Exception("Validation error when reentering node")

            if self._cache_write_idx < self._max_cached_io:
                existing_io_description.cached_values[self._cache_write_idx] = io_description.value

    def increment_cache_idx(self):
        if self._cache_write_idx < self._max_cached_io:
            self._cache_write_idx += 1

    def _build_validation_examples(self):
        examples = [("trace",
            [td.value for td in self.inputs],
            tuple(td.value for td in self.outputs))]

        for cache_idx in range(self._cache_write_idx):
            cached_inputs = [td.cached_values[cache_idx] for td in self.inputs]
            cached_outputs = tuple(td.cached_values[cache_idx] for td in self.outputs)
            examples.append((f"cached[{cache_idx}]", cached_inputs, cached_outputs))
        return examples

    def validate_compiled_model(self, rtol: float = 1e-3, atol: float = 1e-5) -> bool:
        if self.compiled_model is None:
            _get_logger().warning(
                f"Model {self.name} does not have a compiled model. Skipping validation.")
            return True

        examples = self._build_validation_examples()
        all_match = True
        for example_label, input_values, source_outputs in examples:
            with torch.no_grad():
                exported_outputs = self.compiled_model(*input_values)

            if not isinstance(exported_outputs, tuple):
                exported_outputs = (exported_outputs,)

            if len(exported_outputs) != len(source_outputs):
                _get_logger().error(
                    f"{self.name} ({example_label}): Output count mismatch - "
                    f"got {len(exported_outputs)}, expected {len(source_outputs)}")
                all_match = False
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
                    num_elements = exported.numel()
                    _get_logger().error(
                        f"{self.name}/{output_name} ({example_label}): NaN/Inf detected!")
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
                        f"{self.name}/{output_name} ({example_label}): Mismatch detected (rtol={rtol}, atol={atol}). Please check {log_path} for more details.")
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

        if all_match:
            num_examples = len(examples)
            _get_logger().info(
                f"  ✓ {self.name} passed validation ({num_examples} example{'s' if num_examples > 1 else ''})")
        return all_match

    def validate_input_and_update_tags(self, input_name, raw_input_name, input_value):
        self.validate_io_and_update_tags(
            input_name, raw_input_name, input_value, self.inputs)

    def validate_output_and_update_tags(self, output_name, raw_output_name, output_value):
        self.validate_io_and_update_tags(
            output_name, raw_output_name, output_value, self.outputs)

    def change_input_name(self, old_name, new_name):
        _get_logger().warning(
            f"changing input name from {old_name} to {new_name} for model {self.name}")
        if old_name == new_name:
            return
        current_input_names = [input.name_str for input in self.inputs]
        if new_name in current_input_names:
            raise Exception(
                f"Error requesting input name change for {self.name}/{old_name}:"
                f" {new_name} is already in use")
        for input in self.inputs:
            if input.name_str == old_name:
                input.change_name(new_name)

    def change_output_name(self, old_name, new_name):
        _get_logger().warning(
            f"changing output name from {old_name} to {new_name} for model {self.name}")
        if old_name == new_name:
            return
        current_output_names = [output.name_str for output in self.outputs]
        if new_name in current_output_names:
            raise Exception(
                f"Error requesting output name change for {self.name}:"
                f" {new_name} is already in use")
        for output in self.outputs:
            if output.name_str == old_name:
                output.change_name(new_name)

    def compile_trace(self, *args):
        raise NotImplementedError(
            f"compile_trace not implemented for {self.__class__.__name__}")
