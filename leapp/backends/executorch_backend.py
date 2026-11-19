#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import contextlib
import os
import shutil
import tempfile
from importlib.metadata import version
from typing import Tuple

import torch

from leapp.backends.export_backend import ExportBackend, prepare_tensors_for_export
from leapp.utils.logging import _get_logger

DELEGATES = ("xnnpack", "cuda")


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


class ExecuTorchProgram:
    """Callable that runs the ``forward`` method of an ExecuTorch ``.pte`` file.

    Each input is moved to the device the program reads it from. Outputs come
    back on the device of the inputs, like :class:`SimplifiedONNXProgram`, so
    nodes on different devices can feed each other. A ``cuda`` program is built
    for GPU 0 of the visible GPUs and is loaded and run there. Owns ``temp_dir``
    when given and removes it on delete.
    """

    def __init__(self, program_path, data_path=None, cuda=False, temp_dir=None):
        self._temp_dir = temp_dir
        self._cuda = cuda
        Runtime = _import_executorch()

        # Without the cache XNNPACK packs a weight once per use, so a layer
        # called in a loop, such as a denoising step, holds a copy per call.
        backend_options = {}
        if Runtime.get().backend_registry.is_available("XnnpackBackend"):
            backend_options["XnnpackBackend"] = {"weight_cache_enabled": True}

        program = Runtime.get().load_program(
            program_path, data_path=data_path, backend_options=backend_options
        )
        meta = program.metadata("forward")
        self.input_devices = [
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
        if len(args) != len(self.input_devices):
            _get_logger().fatal(
                f"Expected {len(self.input_devices)} inputs, got {len(args)}",
                error_type=ValueError,
            )
        output_device = next(
            (arg.device for arg in args if arg.is_cuda), torch.device("cpu")
        )
        inputs = [
            arg.detach().to(device).contiguous()
            for arg, device in zip(args, self.input_devices)
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
    (CPU, the default) or ``"cuda"`` (ExecuTorch's CUDA backend). The CUDA
    backend keeps inputs and outputs in GPU memory unless
    ``backend_params["device_io"]`` is ``False``. An input the program returns
    unchanged, and that no CUDA operator reads, stays in host memory either way.
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
        self.device_io = self.delegate == "cuda" and device_io
        self.data_path = None
        self._program_dir = None
        self._program_data = None

    def get_backend_metadata(self):
        metadata = {
            "torch_version": str(torch.__version__),
            "executorch_version": version("executorch"),
            "delegate": self.delegate,
        }
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

    def _lower(self, exported_program, program_path):
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from executorch.exir.passes.constant_prop_pass import constant_prop_pass

        if self.delegate == "xnnpack":
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )

            partitioner = XnnpackPartitioner()
            compile_config = None
        else:
            from executorch.backends.cuda.cuda_backend import CudaBackend
            from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

            # Name the weights file after the node, so the nodes of one graph do
            # not overwrite each other's file in the shared output directory.
            partitioner = CudaPartitioner(
                [
                    CudaBackend.generate_method_name_compile_spec("forward"),
                    CudaBackend.generate_data_name_compile_spec(self.node_context.name),
                ]
            )
            # The CUDA backend keeps operators whole for its compiler, which the
            # edge dialect check refuses.
            compile_config = EdgeCompileConfig(_check_ir_validity=False)

        # Fold values computed from constants alone first: an op that only builds
        # a constant may be one the delegate lacks, and left in the graph it
        # would split the program and send work back to the CPU.
        program = to_edge_transform_and_lower(
            constant_prop_pass(exported_program.run_decompositions({})),
            partitioner=[partitioner],
            compile_config=compile_config,
        ).to_executorch(self._backend_config())
        with open(program_path, "wb") as f:
            program.write_to_file(f)
        program.write_tensor_data_to_file(os.path.dirname(program_path))

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
        if self.delegate == "cuda":
            if not torch.cuda.is_available():
                _get_logger().fatal(
                    'ExecuTorch delegate "cuda" needs a CUDA GPU, '
                    "but torch.cuda.is_available() is False",
                    error_type=RuntimeError,
                )
            # GPU 0, which the CUDA backend builds the program for;
            # CUDA_VISIBLE_DEVICES picks the physical GPU.
            device = torch.device("cuda", 0)
            # Shapes are fixed here. Running such a program aborts the process with
            # device_io and fails on every call without it.
            for desc in (*self.node_context.inputs, *self.node_context.outputs):
                if 0 in desc.shape:
                    _get_logger().fatal(
                        f"Tensor '{desc.name}' of node '{self.node_context.name}' has no "
                        'elements, which ExecuTorch delegate "cuda" cannot run. '
                        'Export this node with delegate "xnnpack" or another backend.',
                        error_type=RuntimeError,
                    )

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
        # CUDA backend also uses the current GPU.
        gpu = (
            torch.cuda.device(device)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.device("cpu"), gpu:
            self._lower(exported_program, program_path)

        self._program_dir = temp_dir
        data_path = os.path.splitext(program_path)[0] + ".ptd"
        self._program_data = data_path if os.path.exists(data_path) else None
        self.compiled_model = ExecuTorchProgram(
            program_path,
            self._program_data,
            cuda=self.delegate == "cuda",
            temp_dir=temp_dir,
        )
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
        for filename in os.listdir(self._program_dir):
            staged = os.path.join(save_path, f".{filename}.tmp")
            shutil.copy2(os.path.join(self._program_dir, filename), staged)
            os.replace(staged, os.path.join(save_path, filename))
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
            model_path, data_path, cuda=self.delegate == "cuda"
        )
        # Like the TorchScript and torch.export backends, so the nodes of a
        # pipeline share one device; the program moves inputs and outputs as it
        # needs.
        self.runtime_device = self._select_runtime_device()
        self.compiled_module = None
