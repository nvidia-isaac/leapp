# Torch backend plugins

These plugins execute PyTorch artifacts through Triton's backend API. They build
independently of Cargo and install beneath a backend directory selected by
the `LEAPP_TRITON_BACKENDS` environment variable. Shared request handling, model configuration,
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
cmake -S src/engines/triton/backends/torch/nativert -B target/nativert-source \
  -DCMAKE_BUILD_TYPE=Release -DPYTORCH_SOURCE=/path/to/pytorch-v2.12.1 \
  -DTORCH_ROOT="$TORCH_ROOT"
cmake --build target/nativert-source -j4
```

This builds the raw-program dispatcher path without Python bindings, optional
static kernels requiring private ATen symbols, or CUDA Triton/AOTInductor
delegates. Pass the resulting library to the PT2 plugin build:

```sh
cmake -S src/engines/triton/backends/torch/pt2 -B target/pt2-backend \
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
python tests/generators/generate_models.py --output target/triton-fixtures
python tests/generators/generate_pt2_models.py --output target/pt2-fixtures
cargo test --locked engines::triton::tests::nativert -- --ignored --test-threads=1
cargo test --locked public_pt2 -- --ignored --test-threads=1
```

Generate fixtures with matching PyTorch and an available CUDA GPU. The optional
checks cover weights, noncontiguous outputs, input mutation, concurrent requests,
dynamic and empty tensors, error recovery, integer tensors, incompatible archive
devices, and ONNX/PT2 ensembles. `LEAPP_TRITON_FIXTURES` and `LEAPP_PT2_FIXTURES`
override fixture directories.

## TorchScript

`torch/jit` executes scripted and traced modules with positional tensor arguments
and a tensor or flat tuple of tensors as output. Triton ports use `INPUT__N` and
`OUTPUT__N`; the runtime maps them to public bundle names. Each instance loads
its own module and owns a CUDA stream when executing on a GPU.

Input alias analysis permits borrowing only for read-only arguments. Arguments
that may be mutated or retained, nested subgraphs, and unsupported analysis use
owned storage. Fragmented buffers and transfers to a different execution device
also use owned storage. Outputs are copied into response buffers and stream work
finishes before request release. Session reset restores feedback and defaults;
it does not reset mutable state inside a model.

```sh
cmake -S src/engines/triton/backends/torch/jit -B target/torchscript-backend \
  -DCMAKE_BUILD_TYPE=Release -DTRITON_ROOT="$TRITON_ROOT" \
  -DTORCH_ROOT="$TORCH_ROOT" -DBUILD_TESTING=ON
cmake --build target/torchscript-backend -j4
ctest --test-dir target/torchscript-backend --output-on-failure
./target/torchscript-backend/torchscript_backend_test cuda
cmake --install target/torchscript-backend --prefix "$PWD/target/plugins"
python tests/generators/generate_torchscript_models.py --output target/torchscript-fixtures
cargo test --locked engines::triton::tests::torchscript -- --ignored --test-threads=1
```

The installed library is
`target/plugins/backends/leapp_torchscript/libtriton_leapp_torchscript.so`.
Use the same backend directory and loader setup as the PT2 plugin. Integration
checks require ONNX Runtime and PT2 for mixed ensembles. They cover positional
ports, mutation isolation across inputs and ensemble consumers, feedback after a
failed mutation, dynamic shapes, integer tensors, nested subgraphs, and backend
coexistence. `LEAPP_TORCHSCRIPT_FIXTURES` overrides the fixture directory.
