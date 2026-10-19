// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load and inspect an immutable LEAPP bundle without initializing a backend.
//!
//! ```no_run
//! use leapp_runtime::Bundle;
//! let bundle = Bundle::open("export/policy.yaml")?;
//! for input in bundle.pipeline_inputs() {
//!     if let Some(spec) = bundle.input_info(&input.node, &input.port) {
//!         println!("{}: {:?}", input.name(), spec.shape);
//!     }
//! }
//! # Ok::<(), leapp_runtime::BundleError>(())
//! ```
//! Implementation schema types are intentionally private:
//!
//! ```compile_fail
//! use leapp_runtime::{BundleDescription, NodeSpec, Pipeline};
//! ```
#![forbid(unsafe_code)]

mod bundle;
mod error;

pub use bundle::{Bundle, DataType, Metadata, MetadataValue, PortRef, PortSpec};
pub use error::{BundleError, ErrorCode};
