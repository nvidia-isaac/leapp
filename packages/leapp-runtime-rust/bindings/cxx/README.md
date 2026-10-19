# C++ bindings

C++20 facade over the Rust runtime, using a private cxx bridge. Public headers
expose standard C++ types and status returns. The generated cxx headers are
build artifacts and are not installed.

The Rust core owns parsing, validation, conversion, inference,
feedback, and tensor storage. `bindings/common` shares session locking and explicit
runtime closure with Python. The C++ facade translates descriptors and retains
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
target_link_libraries(my_application PRIVATE leapp::runtime)
```

Set `CMAKE_PREFIX_PATH` to the installation prefix. C++ exceptions are not required.
Operational failures return `Status` with code, phase, and message. Allocation
failure and Rust panics are not recoverable status errors.

## API and ownership

```cpp
leapp::Bundle bundle;
auto status = leapp::Bundle::Open("exports/policy.yaml", &bundle);
if (!status.ok()) return status;

// Inspect metadata before initializing model backends.
const auto* action_info = bundle.output_info("policy", "action");

leapp::Runtime runtime;
status = leapp::Runtime::Open(bundle, {}, &runtime);
if (!status.ok()) return status;

leapp::Session session;
status = runtime.CreateSession(&session);
if (!status.ok()) return status;

std::vector<float> obs(18);
leapp::RunResult result;
status = session.Run(
    {{"preprocess/obs", leapp::TensorView::cpu<float>(obs, {18})}},
    {}, &result);
if (!status.ok()) return status;

leapp::Tensor action;
status = result.GetTensor("policy/action", &action);
if (!status.ok()) return status;
std::span<const float> values;
status = action.CpuSpan(&values);
if (!status.ok()) return status;
```

- Open and CreateSession require empty destinations. Reset a handle through
  assignment to a default instance before reusing it. Failed operations preserve
  destinations; successful Run/GetTensor replace them.
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
- Runtime and Session are movable, noncopyable. Session operations serialize;
  separate sessions can execute concurrently. Close waits for native calls,
  disables all associated sessions, and releases models. Poisoned locks are
  recovered for cleanup, then reported as an error; subsequent Close succeeds.
  Destruction alone permits existing sessions to retain the runtime.
  `runtime.bundle()` exposes the retained bundle snapshot, including after Close.
- Run supplies all public inputs on each call. TensorView owns its shape but
  borrows the allocation. Input storage remains alive and unmodified through the
  call; its byte count must match shape/dtype exactly.
- RunResult and Tensor copies share immutable outputs. They survive subsequent
  runs, reset, and close. A CPU span borrows its tensor owner; retain a Tensor
  while using it. Raw output addresses are read-only by contract.
- GetTensor, metadata getters, and CopyToCpu preserve output-pool eligibility.
  Calling data(), view(), or a successful CpuSpan() permanently excludes that
  allocation from reuse; it does not extend pointer lifetime.
- CpuSpan rejects mismatched types and CUDA storage. Boolean/half formats use
  packed bytes through CopyToCpu; primitive typed helpers cover integers,
  float, and double.

## CUDA buffers

Execution placement and storage placement are independent. Set
`RuntimeOptions.device = ExecutionDevice::kCuda` for GPU model execution.
Specify `RunOptions.output_location` or individual `output_overrides` for outputs.

Construct a TensorView with `MemoryLocation{MemoryType::kCuda, device_id}` and a
device pointer to borrow an existing allocation. Finish producer work before
Run, including work on nondefault streams. No CUDA pointers are dereferenced
on the host. Output storage is ready when Run returns; CopyToCpu performs an
explicit synchronized transfer.

## Validation

Generate fixtures with `tests/generate_models.py`. Configure with
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
