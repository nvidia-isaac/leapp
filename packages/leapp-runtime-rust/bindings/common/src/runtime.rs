// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use leapp_runtime::{
    Bundle, ErrorCode, RunOptions, RunResult, Runtime, RuntimeError, RuntimeOptions, Session,
    TensorView,
};
use std::{
    fmt,
    sync::{Arc, Mutex, RwLock, Weak},
};

/// Binding lifecycle failures preserve native execution diagnostics.
#[derive(Debug)]
pub enum Error {
    Closed,
    Lock,
    Runtime(RuntimeError),
}

impl Error {
    /// Category shared with the core runtime.
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Closed => ErrorCode::InvalidArgument,
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
            Self::Closed => f.write_str("runtime is closed"),
            Self::Lock => f.write_str("runtime lock poisoned"),
            Self::Runtime(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

type SessionSlot = Mutex<Option<Session>>;

struct State {
    bundle: Bundle,
    // Session operations hold a read guard; close waits with the write guard.
    runtime: RwLock<Option<Runtime>>,
    sessions: Mutex<Vec<Weak<SessionSlot>>>,
}

/// Shared runtime with explicit closure of all associated sessions.
#[derive(Clone)]
pub struct RuntimeHandle(Arc<State>);

/// Independent session that retains its runtime and serializes operations.
pub struct SessionHandle {
    state: Arc<State>,
    slot: Arc<SessionSlot>,
}

impl RuntimeHandle {
    /// Load models using the existing core runtime.
    pub fn open(bundle: &Bundle, options: RuntimeOptions) -> Result<Self, Error> {
        let runtime = Runtime::open(bundle, options).map_err(Error::Runtime)?;
        Ok(Self(Arc::new(State {
            bundle: bundle.clone(),
            runtime: RwLock::new(Some(runtime)),
            sessions: Mutex::new(Vec::new()),
        })))
    }

    /// Snapshot remains available after close.
    pub fn info(&self) -> &Bundle {
        &self.0.bundle
    }

    /// Check whether explicit close has released the runtime.
    pub fn is_open(&self) -> Result<bool, Error> {
        Ok(self.0.runtime.read().map_err(|_| Error::Lock)?.is_some())
    }

    /// Create independently initialized feedback state.
    pub fn create_session(&self) -> Result<SessionHandle, Error> {
        let guard = self.0.runtime.read().map_err(|_| Error::Lock)?;
        let runtime = guard.as_ref().ok_or(Error::Closed)?;
        let session = runtime.create_session().map_err(Error::Runtime)?;
        let slot = Arc::new(Mutex::new(Some(session)));
        let mut sessions = self.0.sessions.lock().map_err(|_| Error::Lock)?;
        sessions.retain(|s| s.strong_count() != 0);
        sessions.push(Arc::downgrade(&slot));
        Ok(SessionHandle {
            state: self.0.clone(),
            slot,
        })
    }

    /// Wait for active native calls and close all sessions. Repeated calls succeed.
    pub fn close(&self) -> Result<(), Error> {
        let mut runtime = self.0.runtime.write().map_err(|_| Error::Lock)?;
        let mut sessions = self.0.sessions.lock().map_err(|_| Error::Lock)?;
        for weak in sessions.drain(..) {
            if let Some(session) = weak.upgrade() {
                session.lock().map_err(|_| Error::Lock)?.take();
            }
        }
        runtime.take();
        Ok(())
    }
}

impl SessionHandle {
    /// Borrow inputs until synchronous execution releases them.
    pub fn run(
        &self,
        inputs: &[(&str, TensorView<'_>)],
        options: &RunOptions,
    ) -> Result<RunResult, Error> {
        let guard = self.state.runtime.read().map_err(|_| Error::Lock)?;
        if guard.is_none() {
            return Err(Error::Closed);
        }
        self.slot
            .lock()
            .map_err(|_| Error::Lock)?
            .as_mut()
            .ok_or(Error::Closed)?
            .run_with_options(inputs, options)
            .map_err(Error::Runtime)
    }

    /// Restore the session's explicit feedback state.
    pub fn reset(&self) -> Result<(), Error> {
        let guard = self.state.runtime.read().map_err(|_| Error::Lock)?;
        if guard.is_none() {
            return Err(Error::Closed);
        }
        self.slot
            .lock()
            .map_err(|_| Error::Lock)?
            .as_mut()
            .ok_or(Error::Closed)?
            .reset()
            .map_err(Error::Runtime)
    }
}
