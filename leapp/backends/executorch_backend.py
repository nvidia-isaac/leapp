#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import contextlib
import inspect
import os
import shutil
import tempfile
from importlib.metadata import version
from typing import Tuple

import torch

from leapp.backends.export_backend import ExportBackend, prepare_tensors_for_export
from leapp.utils.logging import _get_logger

DELEGATES = ("xnnpack", "cuda", "tensorrt")
GPU_DELEGATES = ("cuda", "tensorrt")
_TENSORRT_RESERVED_OPTIONS = {
    "exported_program",
    "inputs",
    "arg_inputs",
    "kwarg_inputs",
    "device",
    "dryrun",
}


def _import_executorch():
    try:
        from executorch.runtime import Runtime
    except ImportError as exc:
        _get_logger().fatal(
            "export_with='executorch' needs ExecuTorch: "
            'pip install "leapp[executorch]" (Python 3.11 or newer)',
            error_type=ImportError,
            cause=exc,
        )
    return Runtime


def _import_tensorrt_runtime():
    # The runtime registers the TensorRT delegate with ExecuTorch on import, and
    # a program holding TensorRT engines cannot load without it.
    try:
        import torch_tensorrt_executorch_runtime  # noqa: F401
    except ModuleNotFoundError as exc:
        # Only a missing runtime gets the install hint. One that is installed but
        # lacks a dependency, or fails to load, keeps its own error.
        if exc.name != "torch_tensorrt_executorch_runtime":
            raise
        _get_logger().fatal(
            'ExecuTorch delegate "tensorrt" needs Torch-TensorRT with its '
            'ExecuTorch runtime: pip install "torch_tensorrt[executorch]>=2.15.0"',
            error_type=ImportError,
            cause=exc,
        )


def _import_torch_tensorrt():
    needs = 'ExecuTorch delegate "tensorrt" needs Torch-TensorRT 2.15.0 or newer'
    install = 'pip install "torch_tensorrt[executorch]>=2.15.0"'
    try:
        import torch_tensorrt
    except ModuleNotFoundError as exc:
        if exc.name != "torch_tensorrt":
            raise
        _get_logger().fatal(f"{needs}: {install}", error_type=ImportError, cause=exc)
    # Check for compile_and_save, which builds the program, not for the version:
    # the early 2.15.0 release candidates lack it but report 2.15.0 too, so pip
    # also keeps them on an upgrade.
    if not hasattr(torch_tensorrt, "compile_and_save"):
        _get_logger().fatal(
            f"{needs}, not an early release candidate: pip uninstall "
            f"torch_tensorrt, then {install}",
            error_type=ImportError,
        )
    return torch_tensorrt


class ExecuTorchProgram:
    """Callable that runs the ``forward`` method of an ExecuTorch ``.pte`` file.

    Each input is moved to the device the program reads it from. Outputs come
    back on the device of the inputs, like :class:`SimplifiedONNXProgram`, so
    nodes on different devices can feed each other. A program with no inputs
    returns its outputs on the GPU when there is one. A program of a GPU
    delegate, loaded with ``cuda=True``, is built for GPU 0 of the visible GPUs
    and is loaded and run there. Owns ``temp_dir`` when given and removes it on delete.
    """

    def __init__(
        self, program_path, data_path=None, cuda=False, tensorrt=False, temp_dir=None
    ):
        self._temp_dir = temp_dir
        self._cuda = cuda
        Runtime = _import_executorch()
        # Only a TensorRT program needs the runtime, which maps TensorRT into the
        # process and fails to import against some ExecuTorch builds.
        if tensorrt:
            _import_tensorrt_runtime()

        # Without the cache XNNPACK packs a weight once per use, so a layer
        # called in a loop, such as a denoising step, holds a copy per call.
        backend_options = {}
        if Runtime.get().backend_registry.is_available("XnnpackBackend"):
            backend_options["XnnpackBackend"] = {"weight_cache_enabled": True}

        program = Runtime.get().load_program(
            program_path, data_path=data_path, backend_options=backend_options
        )
        meta = program.metadata("forward")
        self._input_devices = [
            meta.input_tensor_meta(i).device() for i in range(meta.num_inputs())
        ]
        with self._gpu_context():
            self._method = program.load_method("forward")

    def _gpu_context(self):
        # The CUDA backend sets up and runs on the current device, which the caller
        # may have changed from GPU 0.
        if not self._cuda:
            return contextlib.nullcontext()
        return torch.cuda.device(0)

    def __call__(self, *args):
        if len(args) != len(self._input_devices):
            _get_logger().fatal(
                f"Expected {len(self._input_devices)} inputs, got {len(args)}",
                error_type=ValueError,
            )
        # With no inputs to follow, such as a node whose inputs were all pruned,
        # use the GPU when there is one, as TorchScript and torch.export nodes do.
        fallback = "cuda" if not args and torch.cuda.is_available() else "cpu"
        output_device = next(
            (arg.device for arg in args if arg.is_cuda), torch.device(fallback)
        )
        inputs = [
            arg.detach().to(device).contiguous()
            for arg, device in zip(args, self._input_devices)
        ]
        with self._gpu_context():
            outputs = self._method.execute(inputs)
        return tuple(out.to(output_device) for out in outputs)

    def __del__(self):
        # The loaded program keeps the .pte open, and Windows cannot delete an
        # open file.
        self._method = None
        if getattr(self, "_temp_dir", None) is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)


class ExecuTorchExportBackend(ExportBackend):
    """Export node graphs via ``torch.export`` and ExecuTorch to ``.pte`` artifacts.

    ``backend_params["delegate"]`` picks what runs the graph: ``"xnnpack"``
    (CPU, the default), ``"cuda"`` (ExecuTorch's CUDA backend) or
    ``"tensorrt"`` (Torch-TensorRT engines, with what TensorRT does not take
    sent to the CUDA backend). The two GPU delegates keep inputs and outputs in
    GPU memory unless ``backend_params["device_io"]`` is ``False``. An input the
    program returns unchanged, and that no GPU operator reads, stays in host
    memory either way.
    """

    def __init__(self, node_context, backend_params=None):
        super().__init__(node_context, backend_params)
        self.delegate = self.backend_params.get("delegate", "xnnpack")
        if self.delegate not in DELEGATES:
            _get_logger().fatal(
                f"Unsupported ExecuTorch delegate: {self.delegate}, "
                f"please use one of the following: {list(DELEGATES)}",
                error_type=ValueError,
            )
        device_io = self.backend_params.get("device_io", True)
        if not isinstance(device_io, bool):
            _get_logger().fatal(
                f"device_io must be True or False, got {device_io!r}",
                error_type=TypeError,
            )
        self.device_io = self.delegate in GPU_DELEGATES and device_io
        self.tensorrt_options = self.backend_params.get("tensorrt_options", {})
        if not isinstance(self.tensorrt_options, dict):
            _get_logger().fatal(
                f"tensorrt_options must be a dict, got {self.tensorrt_options!r}",
                error_type=TypeError,
            )
        if self.tensorrt_options and self.delegate != "tensorrt":
            _get_logger().fatal(
                'tensorrt_options apply only to delegate "tensorrt", '
                f'not "{self.delegate}"',
                error_type=ValueError,
            )
        reserved = sorted(self.tensorrt_options.keys() & _TENSORRT_RESERVED_OPTIONS)
        if reserved:
            _get_logger().fatal(
                f"tensorrt_options {reserved} cannot be set: LEAPP chooses the "
                "program, its inputs and the GPU, and always builds the engines",
                error_type=ValueError,
            )
        if self.tensorrt_options.get("release_weights_after_build"):
            _get_logger().fatal(
                "tensorrt_options release_weights_after_build cannot be set: it "
                "frees the model's own weights when they are on the GPU, and the "
                "model and later nodes still use them",
                error_type=ValueError,
            )
        self.data_path = None
        self._program_dir = None
        self._program_data = None

    def get_backend_metadata(self):
        metadata = {
            "torch_version": str(torch.__version__),
            "executorch_version": version("executorch"),
            "delegate": self.delegate,
        }
        if self.delegate == "tensorrt":
            # An engine loads only with the TensorRT it was built with, and each
            # Torch-TensorRT version pins one TensorRT.
            metadata["torch_tensorrt_version"] = version("torch_tensorrt")
        if self.data_path is not None:
            metadata["data_path"] = os.path.basename(self.data_path)
            _, metadata["data_sha256sum"] = self._verify_model_location_and_get_hash(
                self.data_path
            )
        return metadata

    def get_backend_model_type(self):
        return "pte"

    def _backend_config(self):
        from executorch.exir import ExecutorchBackendConfig
        from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
        from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig

        if not self.device_io:
            return ExecutorchBackendConfig()
        return ExecutorchBackendConfig(
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
            # The program does not own its GPU inputs and outputs, and ExecuTorch
            # cannot bind a GPU input to a planned buffer.
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            ),
        )

    def _cuda_partitioner(self):
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

        # Name the weights file after the node, so the nodes of one graph do
        # not overwrite each other's file in the shared output directory.
        return CudaPartitioner(
            [
                CudaBackend.generate_method_name_compile_spec("forward"),
                CudaBackend.generate_data_name_compile_spec(self.node_context.name),
            ]
        )

    def _lower(self, exported_program, program_path):
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from executorch.exir.passes.constant_prop_pass import constant_prop_pass
        from executorch.exir.passes.init_mutable_pass import (
            InitializedMutableBufferPass,
        )

        if self.delegate == "xnnpack":
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )

            partitioner = XnnpackPartitioner()
            compile_config = None
        else:
            partitioner = self._cuda_partitioner()
            # The CUDA backend keeps operators whole for its compiler, which the
            # edge dialect check refuses.
            compile_config = EdgeCompileConfig(_check_ir_validity=False)

        exported_program = exported_program.run_decompositions({})
        config = self._backend_config()
        # Fold values computed from constants alone first: an op that only builds
        # a constant may be one the delegate lacks, and left in the graph it
        # would split the program and send work back to the CPU.
        edge = to_edge_transform_and_lower(
            constant_prop_pass(exported_program),
            partitioner=[partitioner],
            compile_config=compile_config,
        )
        # ExecuTorch leaves a buffer the program still writes after lowering
        # uninitialized unless told to start it from its initial value, which
        # it reads as host memory. The pass matches placeholder names, not buffer names.
        lowered = edge.exported_program()
        signature = lowered.graph_signature
        written = set(signature.buffers_to_mutate.values())
        names = []
        for name, buffer in signature.inputs_to_buffers.items():
            if buffer in written:
                names.append(name)
                # torch.export keeps a persistent=False buffer's value in constants.
                values = (
                    lowered.state_dict
                    if buffer in lowered.state_dict
                    else lowered.constants
                )
                values[buffer] = values[buffer].cpu()
        if names:
            config.passes.append(InitializedMutableBufferPass(names))
        program = edge.to_executorch(config)
        with open(program_path, "wb") as f:
            program.write_to_file(f)
        program.write_tensor_data_to_file(os.path.dirname(program_path))

    def _lower_with_tensorrt(self, exported_program, input_values, program_path):
        torch_tensorrt = _import_torch_tensorrt()

        # torch_tensorrt.dynamo.compile ignores keys it does not know, so a typo or
        # a setting it dropped would change nothing without a word.
        known = inspect.signature(torch_tensorrt.dynamo.compile).parameters
        unknown = sorted(
            key
            for key in self.tensorrt_options
            if key not in known or known[key].kind is inspect.Parameter.VAR_KEYWORD
        )
        if unknown:
            _get_logger().fatal(
                f"tensorrt_options {unknown} are not named settings of "
                "torch_tensorrt.dynamo.compile",
                error_type=ValueError,
            )
        # The core ATen decompositions, which Torch-TensorRT runs too, turn a
        # no-op such as detach or a 1-D transpose into alias.
        decomposed = exported_program.run_decompositions()
        signature = decomposed.graph_signature
        # Torch-TensorRT writes a buffer back outside its engines, where
        # ExecuTorch keeps only its shape, so it starts from zeros, and with
        # device_io the export fails. A write to a parameter is lost.
        written = sorted(
            {
                **signature.buffers_to_mutate,
                **signature.parameters_to_mutate,
            }.values()
        )
        if written:
            # The other ExecuTorch delegates cannot export a parameter write either.
            others = (
                "another backend"
                if signature.parameters_to_mutate
                else 'delegate "cuda" or "xnnpack" or another backend'
            )
            _get_logger().fatal(
                f"Node '{self.node_context.name}' writes {written}, which "
                'ExecuTorch delegate "tensorrt" cannot keep. Export this node with '
                f"{others}.",
                error_type=RuntimeError,
            )
        # Torch-TensorRT removes an alias, and leaves a held value it returns
        # outside the engines on the GPU. ExecuTorch then reads it as host memory
        # and crashes the process.
        held = {
            **signature.inputs_to_buffers,
            **signature.inputs_to_parameters,
            **signature.inputs_to_lifted_tensor_constants,
        }
        returned = []
        for node in decomposed.graph.output_node().args[0]:
            while (
                isinstance(node, torch.fx.Node)
                and node.target == torch.ops.aten.alias.default
            ):
                node = node.args[0]
            if isinstance(node, torch.fx.Node) and node.name in held:
                returned.append(held[node.name])
        returned.sort()
        if returned:
            _get_logger().fatal(
                f"Node '{self.node_context.name}' returns {returned} unchanged, "
                'which ExecuTorch delegate "tensorrt" cannot save. Export this node '
                'with delegate "cuda" or "xnnpack" or another backend.',
                error_type=RuntimeError,
            )
        options = {
            "min_block_size": 1,
            # A TensorRT plugin that needs libtorch could not run in ExecuTorch.
            "fallback_pytorch_plugin_ops": True,
            # Compute float32 in full precision, not in TensorRT's default TF32.
            "disable_tf32": True,
            **self.tensorrt_options,
        }
        # Torch-TensorRT's default timing cache is one file in the system temp
        # directory that every account shares, so keep it private to this build.
        cache_dir = None
        if "timing_cache_path" not in options:
            cache_dir = tempfile.mkdtemp()
            options["timing_cache_path"] = os.path.join(cache_dir, "timing_cache.bin")
        try:
            # Unlike compile and then save, this keeps each built engine in a file
            # until the save, and it can save a program built with
            # require_full_compilation.
            torch_tensorrt.compile_and_save(
                exported_program,
                program_path,
                output_format="executorch",
                arg_inputs=list(input_values),
                save_options={
                    # Replay each engine as a CUDA graph, to cut the launch cost
                    # per call.
                    "use_cuda_graphs": True,
                    # What TensorRT leaves runs on the GPU too, never on a CPU
                    # kernel. Built before the engines, so a bad node name fails
                    # first.
                    "partitioners": [self._cuda_partitioner()],
                    "backend_config": self._backend_config(),
                },
                **options,
            )
        finally:
            if cache_dir is not None:
                shutil.rmtree(cache_dir, ignore_errors=True)

    def compile(self, m: torch.nn.Module = None):
        from torch.export.passes import move_to_device_pass

        _import_executorch()
        if m is None:
            m = self.module_builder()
        m = m.eval()

        input_values = [tensor_desc.value for tensor_desc in self.node_context.inputs]
        # ExecuTorchProgram passes contiguous inputs, and a program exported for
        # other strides, such as a transposed input, reads them wrong.
        input_values = tuple(
            value.contiguous() if isinstance(value, torch.Tensor) else value
            for value in prepare_tensors_for_export(input_values)
        )
        device = torch.device("cpu")
        if self.delegate in GPU_DELEGATES:
            if not torch.cuda.is_available():
                _get_logger().fatal(
                    f'ExecuTorch delegate "{self.delegate}" needs a CUDA GPU, '
                    "but torch.cuda.is_available() is False",
                    error_type=RuntimeError,
                )
            # GPU 0, which the GPU delegates build the program for;
            # CUDA_VISIBLE_DEVICES picks the physical GPU.
            device = torch.device("cuda", 0)
            # Shapes are fixed here. The CUDA backend cannot run such a program:
            # it aborts the process with device_io and fails on every call
            # without it. With "tensorrt", which parts the CUDA backend gets is
            # not known before the build.
            for desc in (*self.node_context.inputs, *self.node_context.outputs):
                if 0 in desc.shape:
                    _get_logger().fatal(
                        f"Tensor '{desc.name}' of node '{self.node_context.name}' has no "
                        f'elements, which ExecuTorch delegate "{self.delegate}" cannot '
                        'run. Export this node with delegate "xnnpack" or another '
                        "backend.",
                        error_type=RuntimeError,
                    )
            if self.delegate == "tensorrt":
                _import_tensorrt_runtime()
                _import_torch_tensorrt()

        export_kwargs = {}
        if "strict" in self.backend_params:
            export_kwargs["strict"] = self.backend_params["strict"]
        exported_program = torch.export.export(m, input_values, **export_kwargs)
        # Move the exported program rather than the module, which can hold the
        # user's own submodules. XNNPACK reads weights as host memory, so a weight
        # captured on the GPU would crash it. With autograd on, a buffer that
        # requires grad (as in a traced module) would become a tensor the CUDA
        # backend cannot copy.
        with torch.no_grad():
            exported_program = move_to_device_pass(exported_program, device)

        temp_dir = tempfile.mkdtemp()
        program_path = os.path.join(temp_dir, f"{self.node_context.name}.pte")
        # Lowering creates tensors on the default device, so keep it on the CPU,
        # as in ExecuTorch's own exports, whatever device the caller set. The
        # GPU delegates also build on the current GPU.
        gpu = (
            torch.cuda.device(device)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        # Remove temp_dir if lowering or loading fails.
        try:
            with torch.device("cpu"), gpu:
                if self.delegate == "tensorrt":
                    self._lower_with_tensorrt(
                        exported_program, input_values, program_path
                    )
                else:
                    self._lower(exported_program, program_path)
            data_path = os.path.splitext(program_path)[0] + ".ptd"
            program_data = data_path if os.path.exists(data_path) else None
            compiled_model = ExecuTorchProgram(
                program_path,
                program_data,
                cuda=self.delegate in GPU_DELEGATES,
                tensorrt=self.delegate == "tensorrt",
                temp_dir=temp_dir,
            )
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        self._program_dir = temp_dir
        self._program_data = program_data
        self.compiled_model = compiled_model
        self.compiled_module = m
        return self.compiled_model

    def save(self, save_path: str) -> Tuple[str, str, str]:
        if self._program_dir is None or not os.path.isdir(self._program_dir):
            _get_logger().fatal(
                f"No ExecuTorch program found for {self.node_context.name}",
                error_type=RuntimeError,
            )

        # Copy every file the program was written as, such as the CUDA
        # backend's weights file beside the .pte. Each is renamed into place,
        # since a program still loaded from save_path keeps reading its file.
        # Renaming starts only once all are copied, so a failed copy leaves this
        # node's previous files whole.
        staged = {
            filename: os.path.join(save_path, f".{filename}.tmp")
            for filename in os.listdir(self._program_dir)
        }
        try:
            for filename, path in staged.items():
                shutil.copy2(os.path.join(self._program_dir, filename), path)
        except BaseException:
            for path in staged.values():
                if os.path.exists(path):
                    os.remove(path)
            raise
        for filename, path in staged.items():
            os.replace(path, os.path.join(save_path, filename))
        path = os.path.join(save_path, f"{self.node_context.name}.pte")
        self.data_path = (
            os.path.join(save_path, os.path.basename(self._program_data))
            if self._program_data is not None
            else None
        )
        md5sum, sha256sum = self._verify_model_location_and_get_hash(path)
        return path, md5sum, sha256sum

    def load(self, model_path: str, sha256sum: str):
        data_path = None
        checks = [(model_path, sha256sum)]
        # The CUDA backend keeps the weights in the .ptd, so it is checked too.
        if "data_sha256sum" in self.backend_params:
            data_path = os.path.join(
                os.path.dirname(model_path), self.backend_params["data_path"]
            )
            checks.append((data_path, self.backend_params["data_sha256sum"]))
        for path, expected in checks:
            _, actual = self._verify_model_location_and_get_hash(path)
            if actual != expected:
                _get_logger().fatal(
                    f"SHA256 checksum mismatch for {path}: "
                    f"expected {expected}, got {actual}",
                    error_type=ValueError,
                )

        self.compiled_model = ExecuTorchProgram(
            model_path,
            data_path,
            cuda=self.delegate in GPU_DELEGATES,
            tensorrt=self.delegate == "tensorrt",
        )
        # Like the TorchScript and torch.export backends, so the nodes of a
        # pipeline share one device; the program moves inputs and outputs as it
        # needs.
        self.runtime_device = self._select_runtime_device()
        self.compiled_module = None
