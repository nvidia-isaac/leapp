# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from leapp.backends.export_backend import NoneExportBackend
from leapp.utils.logging import _get_logger


def _import_tensorrt():
    try:
        import tensorrt as trt
    except (ImportError, OSError) as exc:
        _get_logger().fatal(
            "Running a TensorRT engine requires the TensorRT Python package. "
            "Install the TensorRT version and CUDA variant required by your engine.",
            error_type=ImportError,
            cause=exc,
        )
    return trt


class TensorRTProgram:
    """Run a prebuilt engine with fixed, nonempty, linear device tensors.

    Arguments and results follow engine I/O order. Calls synchronize the current
    CUDA stream before returning, so temporary input buffers remain alive until
    inference completes. The program owns one execution context; calls must not
    overlap.
    """

    def __init__(self, model_path):
        trt = _import_tensorrt()
        if not torch.cuda.is_available():
            raise RuntimeError("Running a TensorRT engine requires a CUDA-enabled GPU")

        self.device = torch.device("cuda", torch.cuda.current_device())
        self._logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._logger, "")
        self._runtime = trt.Runtime(self._logger)
        with torch.cuda.device(self.device), open(model_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
            if self._engine is None:
                raise RuntimeError(
                    f"Could not deserialize TensorRT engine {model_path}. Check the "
                    "TensorRT version, GPU compatibility, and required plugins."
                )
            self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Could not create a TensorRT execution context")

        self.inputs = []
        self.outputs = []
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        dtype_map = {
            getattr(trt, name): dtype
            for name, dtype in dtypes.items()
            if hasattr(trt, name)
        }
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            if any(dim <= 0 for dim in shape):
                raise ValueError(
                    f"TensorRT tensor {name!r} must have a nonempty fixed shape, "
                    f"got {shape}"
                )
            if (
                self._engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                or self._engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
            ):
                raise ValueError(
                    f"TensorRT tensor {name!r} must use linear device storage"
                )
            dtype = dtype_map.get(self._engine.get_tensor_dtype(name))
            if dtype is None:
                raise TypeError(f"Unsupported TensorRT dtype for tensor {name!r}")
            tensors = (
                self.inputs
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                else self.outputs
            )
            tensors.append((name, shape, dtype))

    def __del__(self):
        # TensorRT requires the context to die before its engine, and the engine
        # before its runtime. Also handles partially initialized programs.
        self._context = None
        self._engine = None
        self._runtime = None

    def __call__(self, *args):
        if len(args) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(args)}")
        for tensor, (name, shape, dtype) in zip(args, self.inputs):
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
                raise TypeError(f"TensorRT input {name!r} requires a {dtype} tensor")
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"TensorRT input {name!r} requires shape {shape}, "
                    f"got {tuple(tensor.shape)}"
                )

        output_device = next(
            (tensor.device for tensor in args if tensor.is_cuda),
            torch.device("cpu") if args else self.device,
        )
        with torch.cuda.device(self.device):
            inputs = [tensor.detach().to(self.device).contiguous() for tensor in args]
            outputs = [
                torch.empty(shape, dtype=dtype, device=self.device)
                for _, shape, dtype in self.outputs
            ]
            for (name, _, _), tensor in zip(
                self.inputs + self.outputs, inputs + outputs
            ):
                if not self._context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"Could not bind TensorRT tensor {name!r}")
            stream = torch.cuda.current_stream(self.device)
            try:
                if not self._context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT engine execution failed")
            finally:
                # TensorRT borrows these pointers; keep all buffers alive until
                # queued work finishes, including when execution reports failure.
                stream.synchronize()
            return tuple(tensor.to(output_device) for tensor in outputs)


class TensorRTBackend(NoneExportBackend):
    """Load existing TensorRT artifacts without compiling or rebuilding them."""

    def get_backend_model_type(self):
        return "trt"

    def load(self, model_path: str, sha256sum: str, model_type=None):
        _, actual_sha256sum = self._verify_model_location_and_get_hash(model_path)
        if actual_sha256sum != sha256sum:
            _get_logger().fatal(
                f"SHA256 checksum mismatch for {model_path}: "
                f"expected {sha256sum}, got {actual_sha256sum}",
                error_type=ValueError,
            )
        self.compiled_model = TensorRTProgram(model_path)
        self.runtime_device = self.compiled_model.device
