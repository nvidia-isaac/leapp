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

The build command runs CMake to find and validate existing SDKs, then builds both
adapters under the standard user cache. Before replacing an existing build, it
validates the SDKs in a temporary CMake directory; failed validation leaves the
cached build untouched. It never installs dependencies:

- Linux: `build/libleapp_wrp_onnx_custom_op.so` and
  `build/libleapp_wrp_torch_custom_op.so`
- Windows: `build/leapp_wrp_onnx_custom_op.dll` and
  `build/leapp_wrp_torch_custom_op.dll`

LEAPP discovers the cached libraries automatically. Use
`LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY` and
`LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY` to select other build or installation paths.
By default, SDK discovery uses installed CUDA, CMake's Torch search paths, and
Python packages from the interpreter running the helper.

Pass CMake configuration after `--` to select explicit SDKs:

```bash
leapp-build-warp-runtime --build-dir /path/to/build -- \
  -DWARP_ROOT=/path/to/warp \
  -DTorch_DIR=/path/to/libtorch/share/cmake/Torch \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DORT_INCLUDE_DIR=/path/to/onnxruntime/include
```

`WARP_ROOT` contains `native/` and `bin/`; `ORT_INCLUDE_DIR` contains the ONNX
Runtime C API headers. Explicit SDK locations take precedence over Python
package discovery. Use Warp headers, `warp.so`, and exported graphs from the
same Warp release, and a Torch SDK matching the process that loads the adapter.

When no ONNX header directory is supplied, CMake downloads the release selected
by `-DLEAPP_WARP_ONNXRUNTIME_VERSION=VERSION`, defaulting to Python's ONNX Runtime
version. Downloads are cached by version. The adapter does not link another
ONNX Runtime library; its host supplies the API.

Install the adapters with standard CMake commands:

```bash
cmake --install /path/to/build --prefix /path/to/install
```

Libraries install under `CMAKE_INSTALL_LIBDIR` (`GNUInstallDirs`). Warp, Torch,
and CUDA remain in their SDK locations. Build outputs use the discovered library
paths; installed Linux adapters default to `$ORIGIN`. Configure
`CMAKE_INSTALL_RPATH` or set `LD_LIBRARY_PATH` to the selected SDK library
directories when loading installed adapters. Only the LEAPP adapters are installed.

ONNX registers CPU and CUDA kernels. CUDA replay keeps data tensors on the GPU
and uses ONNX Runtime's compute stream; only the embedded archive is copied to
host once during loading. CPU replay stages data through CUDA device 0.
Replay currently supports CUDA device 0 only and restores the caller's CUDA
context. Calls sharing a graph are ordered across streams; warm each instance
serially before concurrent use. Torch makes strided inputs contiguous before
replay.

Torch caches runners without retaining model constants. Expired entries are
released on subsequent calls; native model owners can call
`LEAPP_WARP_ReleaseUnusedRunners()` after releasing constants to clean up
immediately (zero indicates success). Destruction waits for GPU work before
freeing graph resources. Bundle tensors must remain immutable during inference.

Windows builds also require a Warp import library. The `warp-lang` wheel does
not currently include `warp.lib`; set `LEAPP_WARP_IMPORT_LIBRARY` to an
externally supplied import library before building.
