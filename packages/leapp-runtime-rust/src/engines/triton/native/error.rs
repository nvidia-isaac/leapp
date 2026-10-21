// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::sys;
use std::{
    ffi::{CStr, CString},
    fmt,
};

/// Failure category at the native boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidArgument,
    NotFound,
    Unsupported,
    Internal,
}

/// Owned native diagnostic with its operation phase.
#[derive(Debug)]
pub struct Error {
    pub(crate) kind: ErrorKind,
    pub(crate) phase: &'static str,
    pub(crate) message: String,
}

/// Result of a native engine operation.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn new(kind: ErrorKind, phase: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            phase,
            message: message.into(),
        }
    }

    /// Failure category.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Operation that failed.
    pub fn phase(&self) -> &'static str {
        self.phase
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Error {}

pub(crate) fn cstring(value: impl AsRef<[u8]>, phase: &'static str) -> Result<CString> {
    CString::new(value.as_ref()).map_err(|_| {
        Error::new(
            ErrorKind::InvalidArgument,
            phase,
            "embedded NUL in name or path",
        )
    })
}

// Consumes the error objects returned by the pinned Triton API.
pub(crate) unsafe fn check(error: *mut sys::TRITONSERVER_Error, phase: &'static str) -> Result<()> {
    if error.is_null() {
        return Ok(());
    }
    // SAFETY: callers transfer a live, owned Triton error. Copy its diagnostic before deletion.
    unsafe {
        let kind = match sys::TRITONSERVER_ErrorCode(error) {
            sys::TRITONSERVER_ERROR_INVALID_ARG => ErrorKind::InvalidArgument,
            sys::TRITONSERVER_ERROR_NOT_FOUND => ErrorKind::NotFound,
            sys::TRITONSERVER_ERROR_UNSUPPORTED => ErrorKind::Unsupported,
            _ => ErrorKind::Internal,
        };
        let message = CStr::from_ptr(sys::TRITONSERVER_ErrorMessage(error))
            .to_string_lossy()
            .into_owned();
        sys::TRITONSERVER_ErrorDelete(error);
        Err(Error::new(kind, phase, message))
    }
}

pub(crate) struct Handle<T> {
    pub(crate) ptr: *mut T,
    delete: unsafe extern "C" fn(*mut T) -> *mut sys::TRITONSERVER_Error,
}

impl<T> Handle<T> {
    pub(crate) fn empty(
        delete: unsafe extern "C" fn(*mut T) -> *mut sys::TRITONSERVER_Error,
    ) -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            delete,
        }
    }

    pub(crate) fn release(&mut self) -> *mut T {
        std::mem::replace(&mut self.ptr, std::ptr::null_mut())
    }
}

impl<T> Drop for Handle<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: each handle owns one native object and its matching deleter.
            unsafe {
                let _ = check((self.delete)(self.ptr), "release");
            }
        }
    }
}

impl From<Error> for crate::RuntimeError {
    fn from(error: Error) -> Self {
        let code = match error.kind() {
            ErrorKind::InvalidArgument => crate::ErrorCode::InvalidArgument,
            ErrorKind::NotFound => crate::ErrorCode::NotFound,
            ErrorKind::Unsupported => crate::ErrorCode::Unsupported,
            ErrorKind::Internal => crate::ErrorCode::Internal,
        };
        Self::classified(code, error.phase(), error.into())
    }
}
