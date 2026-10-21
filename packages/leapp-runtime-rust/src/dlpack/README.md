<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DLPack API types

This module supplies the backend-independent tensor descriptors and ownership
contracts used by `Session::run` and `RunResult`. It has no native SDK dependency.

`BorrowedTensor` retains a descriptor borrow for synchronous execution.
`ManagedTensor` owns a producer lease and invokes its deleter exactly once.
`SharedTensor` retains a thread-safe allocation and can export managed leases.
The CPU helpers support checked typed borrows and copies of host-accessible data.

The vendored DLPack header, generated Rust ABI, and generator are kept together.
Existing tests cover ABI layout, CPU views, and lease ownership. Native allocation,
CUDA producer handoff, and Python/C++ protocol integration arrive with execution.
