// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use leapp_runtime::dlpack::BorrowedTensor;
use leapp_runtime::{Bundle, ErrorCode, RunResult, RuntimeError, Session, SessionOptions};
use std::{fmt, sync::Mutex};

/// Binding lifecycle failures preserve native execution diagnostics.
/// Wrapped runtime failures remain accessible through [`std::error::Error::source`].
#[derive(Debug)]
pub enum Error {
    Lock,
    Runtime(RuntimeError),
}

impl Error {
    /// Category shared with the core runtime.
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Lock => ErrorCode::Internal,
            Self::Runtime(e) => e.code(),
        }
    }

    /// Operation that failed.
    pub fn phase(&self) -> &str {
        match self {
            Self::Runtime(e) => e.phase(),
            _ => "execute",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lock => f.write_str("session lock poisoned"),
            Self::Runtime(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(error) => Some(error),
            Self::Lock => None,
        }
    }
}

/// Serializes one session's operations and independent closure.
pub struct SessionHandle {
    bundle: Bundle,
    session: Mutex<Session>,
}
impl SessionHandle {
    pub fn open(bundle: &Bundle, options: SessionOptions) -> Result<Self, Error> {
        let session = Session::open(bundle, options).map_err(Error::Runtime)?;
        Ok(Self {
            bundle: bundle.clone(),
            session: Mutex::new(session),
        })
    }
    pub fn info(&self) -> &Bundle {
        &self.bundle
    }
    pub fn is_open(&self) -> Result<bool, Error> {
        Ok(self.session.lock().map_err(|_| Error::Lock)?.is_open())
    }
    pub fn new_session(&self) -> Result<Self, Error> {
        let session = self
            .session
            .lock()
            .map_err(|_| Error::Lock)?
            .new_session()
            .map_err(Error::Runtime)?;
        Ok(Self {
            bundle: self.bundle.clone(),
            session: Mutex::new(session),
        })
    }
    /// Wait for this session's current operation. Recover poison only for cleanup.
    pub fn close(&self) -> Result<(), Error> {
        let mut poisoned = false;
        let mut session = self.session.lock().unwrap_or_else(|error| {
            poisoned = true;
            error.into_inner()
        });
        session.close();
        self.session.clear_poison();
        if poisoned {
            Err(Error::Lock)
        } else {
            Ok(())
        }
    }
    pub fn run(&self, inputs: &[(&str, BorrowedTensor<'_>)]) -> Result<RunResult, Error> {
        self.session
            .lock()
            .map_err(|_| Error::Lock)?
            .run(inputs)
            .map_err(Error::Runtime)
    }
    pub fn reset(&self) -> Result<(), Error> {
        self.session
            .lock()
            .map_err(|_| Error::Lock)?
            .reset()
            .map_err(Error::Runtime)
    }
}
