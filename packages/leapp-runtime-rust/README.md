<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP Rust bundle API

This package loads and validates exported LEAPP bundles and exposes their
public inspection API in Rust. Bundle loading does not initialize an inference
backend. It provides `Bundle`, `PortRef`, `PortSpec`, `DataType`, `Metadata`,
`MetadataValue`, `BundleError`, and `ErrorCode`.

The loader validates graph connections, artifact paths, optional SHA-256 hashes,
and safetensors initial state. It preserves ordered metadata, nested YAML tags,
duplicate mapping keys, and YAML merges.

## Rust

```rust
use leapp_runtime::Bundle;

let bundle = Bundle::open("export/policy.yaml")?;
for output in bundle.pipeline_outputs() {
    let spec = bundle.output_info(&output.node, &output.port).unwrap();
    println!("{}: {:?} {:?}", output.name(), spec.dtype, spec.shape);
}
# Ok::<(), leapp_runtime::BundleError>(())
```

`Bundle` is an immutable, reference-counted snapshot. Clones share the snapshot;
inspection is safe across threads. Getters preserve declaration order and
unknown metadata. Exporter parameters retain their original paths and backend
names. Internal paths are resolved relative to the YAML, falling back to the
supplied artifact path. Optional SHA-256 hashes are checked before publication.
Files must be available while opening; their bytes are not retained for later
execution.

Bundle loading validates graph ports and connections, artifacts, and optional
initial-value files. It does not check whether an inference engine can execute
the graph. Declared external inputs can remain unresolved, matching the source
implementations' inspection contract. Python represents missing input descriptors as `None`.

The loader normalizes known backend aliases, accepts custom nonempty backend
names, and leaves adapter availability to execution.

## Build and test

Run from this directory with a Rust toolchain:

```sh
cargo test --locked
cargo check --locked --workspace
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

The default workspace member is the core crate. Its inspection tests use opaque
artifact bytes and require no inference SDK.

## Common lifecycle sources

The original shared `RuntimeHandle`, `SessionHandle`, and lifecycle errors are
retained unchanged for integration with either runtime implementation. These
sources depend on concrete execution types and are not a separate compiled
crate in this bundle-only checkout. Three regression tests compile the lifecycle
source against a small test-only runtime API.

See [common lifecycle sources](bindings/common/README.md) for integration and
testing instructions.

## Python bindings

The PyO3 binding exposes bundle inspection, detached metadata and port
descriptors, typed stubs, and structured exceptions. Building it requires a
Rust toolchain and Python 3.10 or newer. See
[Python bindings](bindings/python/README.md) for installation and tests.

This distribution shares the `leapp_runtime` import name with the full runtime
implementations; install it in a separate environment.

## Extraction provenance

This package is the common foundation extracted from the existing Triton-based
and native runtime implementations. Execution engines, sessions, tensor/DLPack
code, graph planners, the Triton converter, and backend adapters remain in those
implementations. The package path and library names match the runtime branches
to support later integration. The extraction has no Triton, CUDA, ONNX Runtime,
PyTorch, or NumPy dependency.
