// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private cxx transport for the public C++ facade.
#![deny(unsafe_op_in_unsafe_fn)]

mod bridge;
mod bundle;
mod dtype;
mod session;

use bridge::ffi;
use lrt::ErrorCode;

fn success() -> ffi::StatusData {
    ffi::StatusData {
        code: 0,
        phase: String::new(),
        message: String::new(),
    }
}

fn error(code: ErrorCode, phase: &str, message: impl ToString) -> ffi::StatusData {
    ffi::StatusData {
        code: code as i32,
        phase: phase.into(),
        message: message.to_string(),
    }
}

fn invalid(message: &str) -> ffi::StatusData {
    error(ErrorCode::InvalidArgument, "binding", message)
}

fn binding_status(result: Result<(), leapp_runtime_bindings::Error>) -> ffi::StatusData {
    match result {
        Ok(()) => success(),
        Err(e) => error(e.code(), e.phase(), &e),
    }
}
