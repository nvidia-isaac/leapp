// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Compile the original lifecycle source against a test-only runtime API.
// The real dependency still supplies Bundle and ErrorCode; no production shim.
extern crate leapp_runtime as bundle_api;
extern crate self as leapp_runtime;

pub use bundle_api::{Bundle, ErrorCode};
use std::{
    fmt,
    sync::{mpsc, Arc, Weak},
    thread,
    time::Duration,
};

#[path = "../src/runtime.rs"]
mod lifecycle;
use lifecycle::{Error, RuntimeHandle};

pub type RuntimeOptions = Arc<()>;
pub type TensorView<'a> = &'a [u8];
pub type RunResult = ();

#[derive(Default)]
pub struct RunOptions(Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>);

pub struct Runtime(Arc<()>);

pub struct Session(Weak<()>);

#[derive(Debug)]
pub struct RuntimeError;

impl RuntimeError {
    pub fn code(&self) -> ErrorCode {
        ErrorCode::Internal
    }

    pub fn phase(&self) -> &str {
        "execute"
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("test runtime error")
    }
}

impl Runtime {
    pub fn open(_: &Bundle, owner: RuntimeOptions) -> Result<Self, RuntimeError> {
        Ok(Self(owner))
    }

    pub fn create_session(&self) -> Result<Session, RuntimeError> {
        Ok(Session(Arc::downgrade(&self.0)))
    }
}

impl Session {
    pub fn run_with_options(
        &mut self,
        _: &[(&str, TensorView<'_>)],
        options: &RunOptions,
    ) -> Result<RunResult, RuntimeError> {
        assert!(
            self.0.upgrade().is_some(),
            "runtime dropped during session lifetime"
        );
        if let Some((entered, resume)) = &options.0 {
            entered.send(()).unwrap();
            resume.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), RuntimeError> {
        assert!(
            self.0.upgrade().is_some(),
            "runtime dropped during session lifetime"
        );
        Ok(())
    }
}

fn runtime() -> (RuntimeHandle, Weak<()>) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.yaml");
    std::fs::write(
        &path,
        "models: {}\npipeline: {inputs: {}, outputs: {}}\nsystem information: {}\n",
    )
    .unwrap();
    let owner = Arc::new(());
    let weak = Arc::downgrade(&owner);
    (
        RuntimeHandle::open(&Bundle::open(path).unwrap(), owner).unwrap(),
        weak,
    )
}

#[test]
fn sessions_keep_runtime_alive_after_the_last_runtime_handle_is_dropped() {
    let (runtime, owner) = runtime();
    let session = runtime.create_session().unwrap();
    drop(runtime);
    assert!(owner.upgrade().is_some());
    session.run(&[], &RunOptions::default()).unwrap();
    session.reset().unwrap();
    drop(session);
    assert!(owner.upgrade().is_none());
}

#[test]
fn close_invalidates_all_handles_and_retains_bundle_info() {
    let (runtime, owner) = runtime();
    let clone = runtime.clone();
    let path = runtime.info().yaml_path().to_owned();
    let sessions = [
        runtime.create_session().unwrap(),
        runtime.create_session().unwrap(),
    ];
    clone.close().unwrap();
    assert!(!runtime.is_open().unwrap());
    assert!(owner.upgrade().is_none());
    assert_eq!(runtime.info().yaml_path(), path);
    assert!(matches!(runtime.create_session(), Err(Error::Closed)));
    for session in sessions {
        assert!(matches!(
            session.run(&[], &RunOptions::default()),
            Err(Error::Closed)
        ));
        let error = session.reset().unwrap_err();
        assert!(matches!(error, Error::Closed));
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
        assert_eq!(error.phase(), "execute");
    }
    runtime.close().unwrap();
}

#[test]
fn close_waits_until_an_active_run_finishes() {
    let (runtime, owner) = runtime();
    let session = Arc::new(runtime.create_session().unwrap());
    let (entered, running) = mpsc::channel();
    let (resume, paused) = mpsc::channel();
    let run = {
        let session = session.clone();
        thread::spawn(move || {
            session
                .run(&[], &RunOptions(Some((entered, paused))))
                .unwrap()
        })
    };
    running.recv_timeout(Duration::from_secs(5)).unwrap();
    let (started, closing) = mpsc::channel();
    let (finished, closed) = mpsc::channel();
    let close = {
        let runtime = runtime.clone();
        thread::spawn(move || {
            started.send(()).unwrap();
            runtime.close().unwrap();
            finished.send(()).unwrap();
        })
    };
    closing.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(matches!(
        closed.recv_timeout(Duration::from_millis(50)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert!(owner.upgrade().is_some());
    resume.send(()).unwrap();
    run.join().unwrap();
    close.join().unwrap();
    assert!(owner.upgrade().is_none());
    assert!(matches!(session.reset(), Err(Error::Closed)));
}
