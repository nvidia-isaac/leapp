# C++ bindings

C++20 facade over the Rust runtime, using a private cxx bridge. Public headers
expose standard C++ types and status returns. The generated cxx headers are
build artifacts and are not installed.

The Rust core owns parsing, validation, conversion, inference,
feedback, and tensor storage. `bindings/common` shares session locking and explicit
session closure with Python. The C++ facade translates descriptors and retains
inspection values; it does not implement another runtime.

For native backend authoring, use the separate
[`lrt::native_backend` SDK](../../src/engines/native/backends/README.md#adding-a-backend).
It does not require the C++ consumer facade.

## Build and install

Building on Linux requires Rust 1.88 or newer, CMake 3.22 or newer, and a C++20
compiler. The Triton and CUDA SDKs are not required. Run from
`packages/leapp-runtime-rust`:

```sh
cmake -S bindings/cxx -B target/cxx-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLEAPP_BUILD_TESTING=OFF
cmake --build target/cxx-build -j
cmake --install target/cxx-build --prefix /your/sdk
```

Both engines are included in the same library. Install the model backends
you need as described in the [model support guide](../../README.md#model-support).

CPU, pinned-host, and CUDA buffers are supported by every build. Triton and CUDA
libraries load only when needed; native CPU ONNX with CPU I/O can run without
either. Configure library discovery as described in the
[runtime installation guide](../../README.md#installation).

Consume the installed SDK with:

```cmake
find_package(leapp_runtime CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE lrt::runtime)
```

Set `CMAKE_PREFIX_PATH` to the installation prefix. C++ exceptions are not required.
Operational failures return `Status` with code, phase, and message. Allocation
failure and Rust panics are not recoverable status errors.

## Embed in an application

For a local checkout, add the C++ binding directory and link the same target:

```cmake
add_subdirectory(path/to/leapp/packages/leapp-runtime-rust/bindings/cxx leapp-runtime)
target_link_libraries(my_application PRIVATE lrt::runtime)
```

To fetch a release, keep the full repository layout so Cargo can locate the
core and shared binding crate. Pin `GIT_TAG` to the commit of the runtime release:

```cmake
include(FetchContent)
FetchContent_Declare(leapp
  GIT_REPOSITORY https://github.com/nvidia-isaac/leapp.git
  GIT_TAG <runtime-release-commit>
  SOURCE_SUBDIR packages/leapp-runtime-rust/bindings/cxx)
FetchContent_MakeAvailable(leapp)
target_link_libraries(my_application PRIVATE lrt::runtime)
```

Embedded builds leave LEAPP tests disabled by default. Use
`LEAPP_BUILD_TESTING=ON` to enable them explicitly. Standalone builds also respect
`BUILD_TESTING=OFF`. Debug configurations build Rust's dev profile; other
configurations build its release profile. Ninja Multi-Config is supported.
Build natively on Linux; cross-compilation is not configured. Rust and a C++20
compiler are build dependencies; applications using an installed shared library
do not need Rust at deployment. Pre-1.0 package compatibility and library SONAMEs
are scoped to the minor release.

Use the [Pixi generator](../../packaging/README.md#install-an-application-environment)
with `--binding none` to provision backend dependencies for a source build.

## API and ownership

```cpp
lrt::Bundle bundle;
auto status = lrt::Bundle::Open("exports/policy.yaml", &bundle);
if (!status.ok()) return status;

// Inspect metadata before initializing model backends.
const auto* action_info = bundle.output_info("policy", "action");

lrt::Session session;
lrt::SessionOptions options;
status = lrt::Session::Open(bundle, options, &session);
if (!status.ok()) return status;

std::vector<float> obs(18);
int64_t shape = 18;
DLTensor input{obs.data(), {kDLCPU, 0}, 1, {kDLFloat, 32, 1},
               &shape, nullptr, 0};
lrt::RunResult result;
status = session.Run(
    {{"preprocess/obs", &input}},
    &result);
if (!status.ok()) return status;

lrt::ManagedTensor action;
status = result.GetTensor("policy/action", &action);
if (!status.ok()) return status;
// Inspect action->dl_tensor or transfer action.release() to a DLPack consumer.
// ManagedTensor invokes the standard deleter on destruction.
```

- Open and NewSession require empty destinations. Reset a handle through
  assignment to a default instance before reusing it. Failed operations preserve
  destinations; successful Run/GetTensor replace them.
- `SessionOptions.engine` selects the runtime engine when opening a session.
  The linked runtime defaults to `EngineKind::kNative`. Both engines are
  included; set `engine` explicitly to choose `kTriton` or `kNative`.
  Invalid engine values return `InvalidArgument` without opening an engine or
  changing the destination; there is no fallback.
- Bundle copies share the Rust snapshot and cached C++ inspection values.
  Getter references remain valid while a copy retains that snapshot.
- Missing model/port getters return `nullptr`. Pipeline input/output maps preserve
  declaration order, deduplicate qualified names, and use `std::nullopt` for an
  unresolved descriptor accepted during bundle inspection.
- Metadata uses a variant for null, Boolean, signed/unsigned integers, floating
  point, strings, sequences, and ordered mappings. Mapping keys
  remain typed. `Metadata::Find` handles the common string-key lookup. Custom
  YAML tags are unsupported and fail during parsing.
- `model_io(node)` returns a retained `ModelIO` with input and output names,
  including internal ports. Names are unqualified and preserve declaration order
  and duplicates. Missing models return `nullptr`; empty models return empty
  vectors. References follow the bundle snapshot lifetime.
- Session is movable and noncopyable. Operations on one session serialize;
  separate sessions may be called concurrently. Native peers share models/weights
  and own separate fixed buffers, bindings, and execution contexts. ONNX peers
  can overlap; graphs containing Torch backends or optional ONNX custom operators
  retain a shared execution guard. `NewSession` shares loaded
  models and construction options but starts from captured initial state.
  Close waits for that session's active call and releases its engine reference;
  peers remain usable. Poisoned locks are recovered for cleanup, then reported
  as an error; subsequent Close succeeds. `session.bundle()` exposes the retained
  bundle snapshot, including after Close.
- Run borrows standard `DLTensor` descriptors and their metadata/storage for the
  entire synchronous call. Inputs must be contiguous, ready, and unmodified.
  Byte offsets are honored. The caller guarantees allocation bounds.
- `GetTensor` returns a fresh DLPack 1.0 managed lease. `ManagedTensor` is a
  `unique_ptr<DLManagedTensorVersioned, DLPackDeleter>`; `release()` transfers the
  lease and its deleter responsibility to another consumer. The SDK installs
  the pinned upstream header under `dlpack/dlpack.h`.
- Results and leases survive later runs, reset, and close. Repeated exports may
  share writable storage. Public outputs that also feed back are detached from
  private session state at the configured public location. Never mutate inputs concurrently
  with execution. There are no LEAPP tensor accessors or copy helpers. Allocation
  belongs to each returned result.

`Run(inputs, &result, outputs)` writes named outputs into caller-owned buffers.
Native binds these buffers directly; Triton copies ensemble results into them.
Only omitted outputs are allocated and returned. Buffers must match the
output dtype and shape, meet backend device/alignment requirements, and be disjoint
from inputs and each other. They remain borrowed until return. Their contents are
unspecified on failure. Calling `Run` without output buffers returns fresh storage.

## CUDA buffers

Execution placement and storage placement are independent. Set
`SessionOptions.device = ExecutionDevice::kCuda` for GPU model execution.
Set `SessionOptions.output_location` or individual `output_overrides` at construction.
`Run` takes only inputs and a result destination. Configure the Triton installation
before starting the application, as described
in the [runtime guide](../../README.md#synchronous-execution).

Construct a `DLTensor` with `DLDevice{kDLCUDA, device_id}` and a device pointer.
Finish producer work before Run, including work on nondefault streams. No CUDA
pointers are dereferenced on the host. Output storage is ready when Run returns;
use CUDA or a consuming framework for copies and device transfers.

## Validation

Generate fixtures with `tests/generators/generate_models.py`. Configure with
`BUILD_TESTING=ON`, `LEAPP_FIXTURES` pointing to the generated directory, and
`LEAPP_TEST_BACKENDS` pointing to the installed backends. Set
`LEAPP_TEST_CUDA=ON` to compile CUDA input/output tests with the CUDA SDK.
Tests default to native execution; set `LEAPP_TEST_ENGINE=triton` to exercise
Triton. Then run:

```sh
cmake --build target/cxx-build -j
ctest --test-dir target/cxx-build --output-on-failure
```

The standalone project under `tests/consumer` verifies that an installed SDK
works without Rust sources or generated bridge headers. Python's existing tests
also exercise the shared lifecycle implementation.
