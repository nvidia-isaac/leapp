// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared synchronous lifecycle for language bindings.
#![forbid(unsafe_code)]
mod session;
pub use session::{Error, SessionHandle};
