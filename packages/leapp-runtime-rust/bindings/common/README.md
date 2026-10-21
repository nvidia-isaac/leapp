<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared lifecycle crate

`src/lib.rs` and `src/session.rs` provide session ownership, locking, and
explicit closure. Wrapped runtime errors expose their source chain, and close
releases that session even when its lock is poisoned. Other sessions keep their
engine references.
Close reports the lock error after cleanup; subsequent close calls succeed.
Normal run/reset operations still fail on poisoned locks.

This crate compiles against the exported `Session`, `SessionOptions`, DLPack
input/result types, and errors in `leapp_runtime`. Execution is not implemented
in this API revision: opening a SessionHandle preserves the core Unsupported
error. The runtime integration supplies execution using these same signatures.
The Python and C++ bindings currently expose bundle inspection only.

`SessionHandle::open()` accepts construction-time `SessionOptions`. `run()`
accepts named borrowed inputs without per-call options. `new_session()` creates
an independent session sharing the loaded engine and immutable options.

The root Cargo test target includes `session.rs` against a small session API
defined only in the test binary. Three tests cover waiting for an active run,
error-source preservation, and cleanup after poisoned locks. Execution behavior
is covered by the full runtime package's core execution tests.

```sh
cargo test --locked --test binding_lifecycle
```

## Session reset

`SessionHandle::reset()` delegates to the execution session. State belongs to
each session. `new_session()` shares its engine and construction options, but
starts from the captured initial state rather than current feedback. Reset
restores its feedback and default inputs to their captured initial values.
Those values are loaded when the session opens:
exported initial tensors are used when present, otherwise zeros are used for
statically shaped state. Dynamic state requires an exported initial tensor with
a concrete shape.

Reset serializes with runs on the same session. It preserves other sessions and
retained results, and it does not reread files, reload models, or reset
backend-internal mutable state. The shared wrapper rejects reset after explicit
session closure.

The crate is compiled against the real public Session API. Tensor-state
restoration requires execution and is outside the lifecycle tests here.
