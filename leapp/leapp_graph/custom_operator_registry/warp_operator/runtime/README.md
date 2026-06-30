# LEAPP Warp Runtime

This directory contains the C++ runtime for `leapp::warp_runner` / `com.nvidia.warp::WrpRunner`.

- `core/` is backend-agnostic WRPB extraction, runtime metadata parsing, and Warp APIC replay.
- `onnx/` adapts ONNX Runtime custom-op tensors and CUDA streams to the core runner.
- `torch/` adapts PyTorch/LibTorch dispatcher calls for TorchScript and `.pt2` exported programs.

The runtime consumes the same versioned `runtime_metadata` JSON emitted by the Python tracer/exporter and the same CPU `uint8` WRPB bundle tensor used by ONNX external data.
