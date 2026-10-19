# C++ bundle API

The C++20 facade uses a private cxx bridge to the Rust loader. Public headers
contain no Rust transport types. Rust, CMake 3.22+, and a C++20 compiler are
required on Linux or macOS; inference SDKs are unnecessary.

From the Rust package directory:

```sh
cmake -S bindings/cxx -B target/cxx -DCMAKE_BUILD_TYPE=Release
cmake --build target/cxx --parallel
ctest --test-dir target/cxx --output-on-failure
cmake --install target/cxx --prefix "$PWD/target/install"
cmake -S bindings/cxx/tests/consumer -B target/consumer \
  -DCMAKE_PREFIX_PATH="$PWD/target/install"
cmake --build target/consumer --parallel
./target/consumer/consumer
```

Consumers use `find_package(leapp_runtime CONFIG REQUIRED)` and link
`leapp::runtime`. Include `leapp/bundle.h` for inspection or `leapp/data_type.h`
for scalar type declarations. Custom YAML tags are unsupported; `Bundle::Open`
returns an invalid-argument status in the `parse` phase.

```cpp
leapp::Bundle bundle;
auto status = leapp::Bundle::Open("export/policy.yaml", &bundle);
if (!status.ok()) {
  // status.code(), status.phase(), and status.message() describe the failure.
  return 1;
}
for (const auto& [name, spec] : bundle.pipeline_outputs()) {
  // Inspect the optional PortSpec and its typed Metadata.
}
for (const auto& model : bundle.model_names()) {
  const auto* io = bundle.model_io(model);
  for (const auto& name : io->inputs) {
    const auto* spec = bundle.input_info(model, name);
    // Inspect the internal input descriptor.
  }
}
```

`model_io(node)` returns a retained `const ModelIO*` with `inputs` and `outputs`
name lists, including internal ports. Names are unqualified; declaration order
and duplicates are preserved. An absent model returns `nullptr`, while a model
with no ports has empty lists.

`Bundle` copies retain a shared immutable snapshot. Getter references remain
valid while any copy owns the snapshot. Concurrent reads are safe; assignment
or destruction of the same handle requires synchronization. `Open` requires an
empty destination and leaves it unchanged on failure.
