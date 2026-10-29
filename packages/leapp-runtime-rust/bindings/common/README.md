<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared lifecycle sources

`src/lib.rs` and `src/session.rs` provide session ownership, locking, and
explicit closure. Wrapped runtime errors expose their source chain, and close
releases that session even when its lock is poisoned. Other sessions keep their
engine references.
Close reports the lock error after cleanup; subsequent close calls succeed.
Normal run/reset operations still fail on poisoned locks.

The `leapp-runtime-bindings` crate compiles these wrappers against the core
runtime. It requires Linux and the Triton/CUDA SDKs.
Language bindings use these handles to share closure and session serialization.

The root Cargo test target includes `session.rs` against a small session API
defined only in the test binary. Three tests cover waiting for an active run,
error-source preservation, and cleanup after poisoned locks. Execution behavior is covered by
the core execution tests.

```sh
cargo test --locked --test binding_lifecycle
```

## Session reset

`SessionHandle::reset()` delegates to the execution session. State belongs to
each session. `new_session()` shares its engine and construction options, but starts
from the captured initial state rather than current feedback. Reset restores its feedback and default inputs to their
captured initial values. Those values are loaded when the session opens:
exported initial tensors are used when present, otherwise zeros are used for
statically shaped state. Dynamic state requires an exported initial tensor with
a concrete shape.

Reset serializes with runs on the same session. It preserves other sessions and
retained results, and it does not reread files, reload models, or reset backend-internal mutable state. The shared
wrapper rejects reset after explicit session closure.

## Inputs and output storage

`run` forwards a borrowed `HashMap<&str, BorrowedTensor<'_>>`; language adapters
retain their owners for the synchronous call. Input map order does not affect
routing. Output declaration order remains stable.

Each core session owns its output pool. Peers share the engine and inherit
`OutputPoolOptions`, but start with separate, empty caches. Closing a handle
releases only its cache references; retained results remain usable. The common
wrapper supplies serialization and closure without allocating tensor buffers.
