// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bundle inspection and the public Session API.
//!
//! Session execution is supplied by the runtime integration; in this API revision,
//! [`Session::open`] returns [`ErrorCode::Unsupported`].
//!
//! Implementation schema types are intentionally private:
//!
//! ```compile_fail
//! use leapp_runtime::{BundleDescription, NodeSpec, Pipeline};
//! ```
//!
//! Engine implementations are an internal compile-time choice:
//!
//! ```compile_fail
//! use leapp_runtime::engines;
//! ```
#![deny(unsafe_op_in_unsafe_fn)]

mod bundle;
pub mod dlpack;
mod engines;
mod error;
mod session;

pub use bundle::{Bundle, DataType, Metadata, MetadataValue, PortRef, PortSpec};
pub use engines::{ExecutionDevice, MemoryLocation};
pub use error::{BundleError, ErrorCode, RuntimeError};
pub use session::{OutputPlacement, RunResult, Session, SessionOptions};
