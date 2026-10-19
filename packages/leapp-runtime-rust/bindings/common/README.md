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

The `leapp-runtime-bindings` crate compiles these wrappers against the core
runtime with its `triton` feature. It requires Linux and the Triton/CUDA SDKs.
Language bindings use these handles to share closure and session serialization.

The root Cargo test target includes `runtime.rs` against a small runtime API
defined only in the test binary. Five tests cover session ownership, close
invalidation and retained bundle info, waiting for an active run, error-source
preservation, and cleanup after poisoned locks. Execution behavior is covered by the core execution tests.

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
