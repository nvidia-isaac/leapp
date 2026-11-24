<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Bazel integration

`packages/leapp-rt` is a self-contained Bzlmod module named `leapp_rt`.
Build natively on Linux with a C++20 compiler. Bazel downloads the declared
Rust toolchain, crates, and code generators; building the runtime requires no
Triton, CUDA, LibTorch, ONNX Runtime, Python, Cargo installation, or CMake.
The default Rust toolchain is pinned to 1.99.0. A consuming project's registered
toolchains take precedence; its Rust toolchain must satisfy the Cargo manifests'
minimum version. Bazel 8.4.2 is pinned for development and CI. Linux x86_64 and
Linux aarch64 are tested.

## Add the module to an application

Until the module is published to a Bazel registry, use a pinned Git override.
Choose a full commit containing the Bazel support, and keep `strip_prefix` so
Bazel loads the runtime module rather than the exporter repository root:

```starlark
# MODULE.bazel
bazel_dep(name = "leapp_rt", version = "0.1.0")
git_override(
    module_name = "leapp_rt",
    remote = "https://github.com/nvidia-isaac/leapp.git",
    commit = "<full-commit-sha>",
    strip_prefix = "packages/leapp-rt",
)
```

For local development, replace `git_override` with:

```starlark
local_path_override(
    module_name = "leapp_rt",
    path = "/path/to/leapp/packages/leapp-rt",
)
```

The module declares its transitive dependencies and default Rust toolchain.
There are no LEAPP dependency-setup macros or required `.bazelrc` settings.
Consumers declare the rulesets they load in their own BUILD files:

```starlark
# Additional entries in the application's MODULE.bazel
bazel_dep(name = "rules_cc", version = "0.2.17")
bazel_dep(name = "rules_rust", version = "0.75.0")
```

```starlark
# BUILD.bazel
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_rust//rust:defs.bzl", "rust_binary")

cc_binary(
    name = "application",
    srcs = ["main.cc"],
    copts = ["-std=c++20"],
    deps = ["@leapp_rt//:runtime"],
)

rust_binary(
    name = "rust_application",
    srcs = ["main.rs"],
    edition = "2021",
    deps = ["@leapp_rt//:lrt"],
)
```

C++ uses `#include "leapp/runtime.h"` and the `lrt` namespace; Rust uses the
`lrt` crate. `runtime` links the shared C++ facade, with both engines included.
Bazel supplies its runtime library through runfiles. Applications distributed
outside Bazel must deploy `libleapp_rt.so` and configure their library search
path. CXX bridge headers and Rust implementation targets are private.

Backend libraries still load dynamically when opening sessions. Provision the
selected model runtimes using the
[runtime installation guide](../../../docs/source/runtime/installation.rst).
Python packaging and optional Torch/NativeRT/Triton plugin builds retain their
existing Maturin/CMake workflows.

## Write a native backend

`@leapp_rt//:native_backend` exposes the public C++ backend headers. The
`native_backend_plugin` helper compiles the adapter source with the plugin's
own compiler flags and STL ABI, rather than sharing an adapter built for a
different model SDK:

```starlark
load("@leapp_rt//bazel:defs.bzl", "native_backend_plugin")

native_backend_plugin(
    name = "libleapp_example.so",
    srcs = ["plugin.cc"],
    # Set SDK-specific flags and dependencies here.
    copts = ["-D_GLIBCXX_USE_CXX11_ABI=0"],
    linkopts = ["-Wl,--no-undefined"],
)
```

The helper builds a `cc_binary(linkshared = True)` and accepts its other
attributes. It adds C++20 for this target only. Plugins export
`LEAPP_NATIVE_BACKEND(example, loader)` as described in the
[backend authoring guide](../AGENTS.md#adding-a-backend).
Build systems with their own plugin rule can place
`@leapp_rt//:native_backend_adapter` in that rule's `srcs` and depend on
`@leapp_rt//:native_backend` directly. The headers target alone does not link
the adapter implementations.

## Build and validate

From `packages/leapp-rt`:

```sh
bazel build //:lrt //bindings/cxx:libleapp_rt.so
bazel test //... --test_output=errors
```

The default tests cover bundle parsing, DLPack ownership and ABI, CPU memory,
shared binding lifecycle, the C++ facade, and backend SDK types. The DLPack ABI
test invokes the host `cc` compiler to compare the pinned C header's layout.

The smoke tests in `tests/bazel` verify downstream Rust/C++ linking and loading
a backend plugin built with a different STL ABI. CI copies the project outside
the repository and consumes a source archive containing only this module,
without a LEAPP bazelrc or consumer-side toolchain registration. For a local
consumer check:

```sh
# From a separate copy of tests/bazel:
bazel test //... --override_module=leapp_rt=/path/to/leapp/packages/leapp-rt
```

Generate ONNX fixtures with `tests/generators/generate_models.py` in an exporter
environment. To exercise CPU ONNX and C++ sessions, configure `ORT_DYLIB_PATH`
and its shared-library dependencies, then run:

```sh
bazel run //:run -- /absolute/path/to/fixtures/policy.yaml
LEAPP_TEST_ENGINE=native bazel run //bindings/cxx:inference_test -- /absolute/path/to/fixtures OFF
```

For native ONNX CPU/CUDA peer execution, supply GPU-enabled ONNX Runtime,
CUDA/cuDNN libraries, a GPU, and an absolute fixture directory. These tests are
explicit opt-ins and fail if enabled dependencies are unavailable:

```sh
bazel test //:native_onnx_test \
  --test_env=LEAPP_TRITON_FIXTURES=/absolute/path/to/fixtures \
  --test_env=ORT_DYLIB_PATH --test_env=CUDA_ROOT --test_env=LD_LIBRARY_PATH \
  --test_output=errors
bazel test //:memory_test --test_env=LEAPP_TEST_CUDA=1 \
  --test_env=CUDA_ROOT --test_env=LD_LIBRARY_PATH --test_output=errors
```

## Update dependencies

The Cargo workspace manifests and `Cargo.lock` remain the source of truth.
`Cargo.Bazel.lock` is the generated crate-universe graph required for transitive
Bzlmod use; `MODULE.bazel.lock` records Bazel module resolution. After changing
Cargo dependencies, regenerate and review both Bazel lockfiles from this module:

```sh
CARGO_BAZEL_REPIN=1 bazel build //:lrt //bindings/cxx:libleapp_rt.so
bazel mod deps
```

The Bazel integration preserves `Cargo.lock` during generation. Keep the
`cxx.rs` module version equal to the `cxx` dependency in `bindings/cxx/Cargo.toml`.
Run Buildifier over changed `.bzl` and `.bazel` files, and rerun both the module
tests and the independent consumer after repinning. The CI script
`.github/scripts/runtime-bazel.sh` performs formatting and both test runs using
temporary build directories that it removes on completion.
