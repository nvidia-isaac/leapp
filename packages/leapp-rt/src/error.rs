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

#[derive(Debug)]
pub(crate) struct Unsupported(pub String);

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unsupported {}

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

#[derive(Debug)]
pub(crate) struct Internal(pub String);

impl fmt::Display for Internal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Internal {}

fn classify_error(source: &anyhow::Error) -> ErrorCode {
    if let Some(error) = source.downcast_ref::<RuntimeError>() {
        return error.code();
    }
    if source.is::<Internal>() {
        return ErrorCode::Internal;
    }
    if let Some(error) = source.downcast_ref::<io::Error>() {
        if error.kind() == io::ErrorKind::NotFound {
            ErrorCode::NotFound
        } else {
            ErrorCode::Io
        }
    } else if source.is::<Unsupported>() {
        ErrorCode::Unsupported
    } else {
        ErrorCode::InvalidArgument
    }
}

/// Execution failure with its phase and underlying diagnostic.
#[derive(Debug)]
pub struct RuntimeError {
    code: ErrorCode,
    phase: &'static str,
    source: anyhow::Error,
}

impl RuntimeError {
    pub(crate) fn new(phase: &'static str, source: anyhow::Error) -> Self {
        Self {
            code: classify_error(&source),
            phase,
            source,
        }
    }

    pub(crate) fn classified(code: ErrorCode, phase: &'static str, source: anyhow::Error) -> Self {
        Self {
            code,
            phase,
            source,
        }
    }

    /// Category suitable for programmatic error handling.
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// Operation that failed, such as `convert`, `load`, `state`, or `execute`.
    pub fn phase(&self) -> &'static str {
        self.phase
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:#}", self.phase, self.source)
    }
}

impl std::error::Error for RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}
