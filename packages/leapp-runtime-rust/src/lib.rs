// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load and inspect an immutable LEAPP bundle without initializing a backend.
//!
//! Implementation schema types are intentionally private:
//!
//! ```compile_fail
//! use lrt::{BundleDescription, NodeSpec, Pipeline};
//! ```
//!
//! Execution uses embedded Triton; engine implementations remain private:
//!
//! ```compile_fail
//! use lrt::engines;
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
pub use session::{OutputPlacement, OutputPoolOptions, RunResult, Session, SessionOptions};

#[cfg(test)]
extern crate self as lrt;
