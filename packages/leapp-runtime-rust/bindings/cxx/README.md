# C++ bindings

C++20 facade over the Rust runtime, using a private cxx bridge. Public headers
expose standard C++ types and status returns. The generated cxx headers are
build artifacts and are not installed.

The Rust core owns parsing, validation, conversion, inference,
feedback, and tensor storage. `bindings/common` shares session locking and explicit
session closure with Python. The C++ facade translates descriptors and retains
inspection values; it does not implement another runtime.

## Build and install

Linux x86-64 requires Rust 1.88 or newer, CMake 3.22 or newer, a C++20 compiler,
the Triton and CUDA SDKs, and the selected backend libraries. Run from `packages/leapp-runtime-rust`:

```sh
cmake -S bindings/cxx -B target/cxx-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRITON_ROOT=/opt/tritonserver \
  -DCUDA_ROOT=/usr/local/cuda \
  -DBUILD_TESTING=OFF
cmake --build target/cxx-build -j
cmake --install target/cxx-build --prefix /your/sdk
```

CPU, pinned-host, and CUDA buffers are supported by every build. Native backend
and SDK libraries remain external dependencies; configure the loader search path
(`LD_LIBRARY_PATH`) for your deployment.

Consume the installed SDK with:

```cmake
find_package(leapp_runtime CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE lrt::runtime)
```

Set `CMAKE_PREFIX_PATH` to the installation prefix. C++ exceptions are not required.
Operational failures return `Status` with code, phase, and message. Allocation
failure and Rust panics are not recoverable status errors.

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
- Triton is selected automatically when opening a session.
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
  separate sessions can execute concurrently. `NewSession` shares the loaded
  engine and construction options but starts from captured initial state.
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
  pooling belongs to each session; exporting a lease excludes its storage from
  reuse.

`SessionOptions.output_pool.max_entries` limits cached buffers (default 64).
Zero disables caching. The byte budget follows bundle output sizes and grows for
larger dynamic outputs; there is no fixed byte cap. `NewSession` inherits the
policy with an empty cache. The limit does not restrict live results, and closing
one session leaves retained outputs and peer sessions valid.

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
`LEAPP_TEST_CUDA=ON` to include CUDA input/output combinations, then run:

```sh
cmake --build target/cxx-build -j
ctest --test-dir target/cxx-build --output-on-failure
```

The standalone project under `tests/consumer` verifies that an installed SDK
works without Rust sources or generated bridge headers. Python's existing tests
also exercise the shared lifecycle implementation.
