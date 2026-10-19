# Torch backend plugins

These plugins execute PyTorch artifacts through Triton's backend API. They build
independently of Cargo and install beneath a backend directory selected by
`RuntimeOptions::backend_directory`. Shared request handling, model configuration,
and tensor I/O live in `torch/common`.

Build commands below run from `packages/leapp-runtime-rust`. Supply Linux,
CMake 3.24+, a C++20 compiler, nlohmann_json 3.11+, Triton 2.69.0, CUDA, and
CUDA-enabled LibTorch with the C++11 libstdc++ ABI. CUDA-enabled LibTorch is
required even for CPU execution. Builds use installed dependencies without
fetching or installing them. Set `TRITON_ROOT` and `TORCH_ROOT` to their SDK roots.

## PT2 / NativeRT

`torch/pt2` executes raw `torch.export.save` archives with tensor-only user inputs
and outputs in exported-program order. Each model instance owns a NativeRT runner
and a CUDA stream when executing on a GPU. Inputs use owned storage to contain
mutation; outputs are copied into Triton's response buffers. CPU, pinned-host,
and CUDA buffers are supported independently of model placement. Copies and
execution complete before the request is released, including error paths.

Archive input devices must match the selected instance. NativeRT does not remap
archive weights or graph devices; host constants in CUDA exports are preserved.
Models use `max_batch_size: 0` and coupled responses.

Supply a NativeRT implementation matching LibTorch. Configuration checks its
headers and linkage. The source-build recipe in `torch/nativert` supports
PyTorch v2.12.1 source with submodules and LibTorch 2.12.1:

```sh
cmake -S src/backends/torch/nativert -B target/nativert-source \
  -DCMAKE_BUILD_TYPE=Release -DPYTORCH_SOURCE=/path/to/pytorch-v2.12.1 \
  -DTORCH_ROOT="$TORCH_ROOT"
cmake --build target/nativert-source -j4
```

This builds the raw-program dispatcher path without Python bindings, optional
static kernels requiring private ATen symbols, or CUDA Triton/AOTInductor
delegates. Pass the resulting library to the PT2 plugin build:

```sh
cmake -S src/backends/torch/pt2 -B target/pt2-backend \
  -DCMAKE_BUILD_TYPE=Release -DTRITON_ROOT="$TRITON_ROOT" \
  -DTORCH_ROOT="$TORCH_ROOT" \
  -DNATIVERT_LIBRARY="$PWD/target/nativert-source/libleapp_nativert.so"
cmake --build target/pt2-backend -j4
cmake --install target/pt2-backend --prefix "$PWD/target/plugins"
```

The installed library is
`target/plugins/backends/leapp_pt2/libtriton_leapp_pt2.so`. Make LibTorch,
NativeRT, Triton, CUDA, and backend dependencies discoverable through the dynamic
loader, for example with `LD_LIBRARY_PATH`.

For integration tests, also make the installed ONNX Runtime backend available:

```sh
ln -s "$TRITON_ROOT/backends/onnxruntime" target/plugins/backends/onnxruntime
export LEAPP_TRITON_BACKENDS="$PWD/target/plugins/backends"
python tests/generate_models.py --output target/triton-fixtures
python tests/generate_pt2_models.py --output target/pt2-fixtures
cargo test --locked --features triton adapter::nativert_tests -- --ignored --test-threads=1
cargo test --locked --features triton --test execution public_pt2 -- --ignored --test-threads=1
```

Generate fixtures with matching PyTorch and an available CUDA GPU. The optional
checks cover weights, noncontiguous outputs, input mutation, concurrent requests,
dynamic and empty tensors, error recovery, integer tensors, incompatible archive
devices, and ONNX/PT2 ensembles. `LEAPP_TRITON_FIXTURES` and `LEAPP_PT2_FIXTURES`
override fixture directories.
