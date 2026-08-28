# LEAPP Warp Runtime

This directory contains the C++ runtime for `leapp::warp_runner` / `com.nvidia.warp::WrpRunner`.

- `core/` is backend-agnostic WRPB extraction, runtime metadata parsing, and Warp APIC replay.
- `onnx/` adapts ONNX Runtime custom-op tensors and CUDA streams to the core runner.
- `torch/` adapts PyTorch/LibTorch dispatcher calls for TorchScript and `.pt2` exported programs.

The runtime consumes the same `runtime_metadata` JSON emitted by the Python tracer/exporter and the same CPU `uint8` WRPB bundle tensor used by ONNX external data.

Installed packages provide two commands:

```bash
leapp-build-warp-runtime
leapp-build-warp-runtime --status
```

The build command validates existing dependencies without installing them,
then builds both adapters under the standard user cache:

- Linux: `build/libleapp_wrp_onnx_custom_op.so` and
  `build/libleapp_wrp_torch_custom_op.so`
- Windows: `build/leapp_wrp_onnx_custom_op.dll` and
  `build/leapp_wrp_torch_custom_op.dll`

LEAPP discovers the cached libraries automatically. Use
`LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY` and
`LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY` only to override the cache paths. Build with
the Python environment that will load the libraries because the adapters link
against that environment's Warp and PyTorch installations.

Windows builds also require a Warp import library. The `warp-lang` wheel does
not currently include `warp.lib`; set `LEAPP_WARP_IMPORT_LIBRARY` to an
externally supplied import library before building.
