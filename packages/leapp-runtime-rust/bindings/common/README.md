<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared lifecycle sources

`src/lib.rs` and `src/runtime.rs` provide session ownership, locking, and explicit
closure.

This source depends on the concrete `Runtime`, `Session`, tensor, options, and
error types in `leapp_runtime`. Those execution types are not part of this
inspection-only package, so this directory is not a separate Cargo crate here.
The Python and C++ inspection bindings use the core bundle crate directly.
Runtime packages compile these sources with their execution types and features.

The root Cargo test target includes `runtime.rs` against a small runtime API
defined only in the test binary. Three tests cover session ownership, close
invalidation and retained bundle info, and waiting for an active run. Execution
behavior is tested in the full runtime integration.

```sh
cargo test --locked --test binding_lifecycle
```
