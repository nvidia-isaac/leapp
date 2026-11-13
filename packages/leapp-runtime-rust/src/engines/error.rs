// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

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
    pub(crate) fn new(
        kind: ErrorKind,
        phase: &'static str,
        message: impl Into<String>,
    ) -> Self {
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
