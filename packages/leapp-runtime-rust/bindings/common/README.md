<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Original shared lifecycle sources

`src/lib.rs` and `src/runtime.rs` are unchanged copies of the common lifecycle
used by both original runtime implementations. They retain session
ownership, locking, explicit closure, and the original error handling.

This source depends on the concrete `Runtime`, `Session`, tensor, options, and
error types in `leapp_runtime`. Those execution types are not part of this
bundle-only checkout, so this directory is not a separate Cargo crate here.
The Python and C++ inspection bindings use the core bundle crate directly.
When integrating with either runtime branch, use these common sources with that
branch's existing common-crate manifest and execution feature.

The root Cargo test target includes the original `runtime.rs` against a small
runtime API defined only in the test binary. Three tests verify that sessions
retain their runtime, explicit close invalidates all handles while preserving
bundle info, and close waits for an active run. These test lifecycle behavior,
not model execution. No adapter traits or cleanup behavior changes are included.

```sh
cargo test --locked --test binding_lifecycle
```
