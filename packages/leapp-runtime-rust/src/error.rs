// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    fmt, io,
    path::{Path, PathBuf},
};

/// Failure categories returned by fallible runtime operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ErrorCode {
    Ok = 0,
    InvalidArgument = 1,
    NotFound = 2,
    Unsupported = 3,
    Io = 4,
    Internal = 5,
}

/// An owned loading error, including its phase, YAML path and underlying cause.
#[derive(Debug)]
pub struct BundleError {
    code: ErrorCode,
    phase: &'static str,
    path: PathBuf,
    source: anyhow::Error,
}

impl BundleError {
    /// Category suitable for programmatic error handling.
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// Loading phase that failed, such as `parse` or `artifact`.
    pub fn phase(&self) -> &'static str {
        self.phase
    }

    /// YAML path supplied to the failed open attempt.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Full contextual diagnostic, including the original dependency error.
    pub fn message(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:#}", self.path.display(), self.source)
    }
}

impl std::error::Error for BundleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

pub(crate) fn in_phase<T>(
    path: &Path,
    phase: &'static str,
    operation: impl FnOnce() -> anyhow::Result<T>,
) -> Result<T, BundleError> {
    operation().map_err(|source| {
        let code = classify_error(&source);
        BundleError {
            code,
            phase,
            path: path.into(),
            source,
        }
    })
}

fn classify_error(source: &anyhow::Error) -> ErrorCode {
    if let Some(error) = source.downcast_ref::<io::Error>() {
        if error.kind() == io::ErrorKind::NotFound {
            ErrorCode::NotFound
        } else {
            ErrorCode::Io
        }
    } else {
        ErrorCode::InvalidArgument
    }
}
