<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared lifecycle sources

`src/lib.rs` and `src/runtime.rs` provide session ownership, locking, and explicit
closure. Wrapped runtime errors expose their source chain, and close finishes
releasing every session and the runtime even when locks are poisoned.
Close reports the lock error after cleanup; subsequent
close calls succeed. Normal run/reset operations still fail on poisoned locks.

This source depends on the concrete `Runtime`, `Session`, tensor, options, and
error types in `leapp_runtime`. Those execution types are not part of this
inspection-only package, so this directory is not a separate Cargo crate here.
The Python and C++ inspection bindings use the core bundle crate directly.
Runtime packages compile these sources with their execution types and features.

The root Cargo test target includes `runtime.rs` against a small runtime API
defined only in the test binary. Five tests cover session ownership, close
invalidation and retained bundle info, waiting for an active run, error-source
preservation, and cleanup after poisoned locks. Execution behavior is tested in
the full runtime integration.

```sh
cargo test --locked --test binding_lifecycle
```

## Session reset

`SessionHandle::reset()` delegates to the execution session. State belongs to
each session, so reset restores its feedback and default inputs to their captured
initial values. Those values are loaded when the runtime opens: exported
initial tensors are used when present, otherwise zeros are used for statically
shaped state. Dynamic state requires an exported initial tensor with a concrete
shape.

Reset serializes with runs on the same session. It preserves other sessions and
retained results, and it does not reread files, reload models, reset
backend-internal mutable state, or flush shared allocation pools. The shared
wrapper rejects reset after explicit runtime closure.

This inspection-only package provides the reset wrapper as source; it does not
export an executable runtime/session API. Tensor-state restoration requires the
execution engine and is outside the lifecycle tests here.
