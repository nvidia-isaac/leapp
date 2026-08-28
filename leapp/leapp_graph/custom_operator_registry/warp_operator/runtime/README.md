# LEAPP Warp Runtime

This directory contains the C++ runtime for `leapp::warp_runner` / `com.nvidia.warp::WrpRunner`.

- `core/` is backend-agnostic WRPB extraction, runtime metadata parsing, and Warp APIC replay.
- `onnx/` adapts ONNX Runtime custom-op tensors and CUDA streams to the core runner.
- `torch/` adapts PyTorch/LibTorch dispatcher calls for TorchScript and `.pt2` exported programs.

The runtime consumes the same `runtime_metadata` JSON emitted by the Python tracer/exporter and the same CPU `uint8` WRPB bundle tensor used by ONNX external data.

The default CMake configuration builds both adapters:

- `build/libleapp_wrp_onnx_custom_op.so`
- `build/libleapp_wrp_torch_custom_op.so`

Use `LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY` for ONNX exports and
`LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY` for `.pt2` exports. Build with the Python
environment that will load the libraries because the adapters link against
that environment's ONNX Runtime and PyTorch installations.
