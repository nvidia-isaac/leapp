// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synchronous, owning adapter for the supplied Triton C library.
//! Requires Linux Triton 2.69.0 and a compatible CUDA SDK.
#![deny(unsafe_op_in_unsafe_fn)]

mod cuda;
mod error;
mod sys;
mod tensor;

pub use error::{Error, ErrorKind, Result};
pub use tensor::{DataType, MemoryLocation, Tensor, TensorElement, TensorView};

pub use cuda::{synchronize_external_stream, HandoffStream};
