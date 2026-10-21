// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Compile the binding lifecycle against a test-only session.
extern crate leapp_runtime as bundle_api;
extern crate self as leapp_runtime;
pub use bundle_api::{Bundle, ErrorCode};
use std::{
    fmt,
    sync::{mpsc, Arc, Weak},
    thread,
    time::Duration,
};

mod lifecycle {
    include!("../src/session.rs");
    pub fn set_wait(
        session: &SessionHandle,
        entered: std::sync::mpsc::Sender<()>,
        resume: std::sync::mpsc::Receiver<()>,
    ) {
        session.session.lock().unwrap().wait = Some((entered, resume));
    }
    #[test]
    fn close_recovers_poison_without_closing_peers() {
        let (session, owner) = crate::session();
        let peer = session.new_session().unwrap();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = session.session.lock().unwrap();
            panic!("simulate session panic");
        }));
        assert!(matches!(session.reset(), Err(Error::Lock)));
        assert!(matches!(session.close(), Err(Error::Lock)));
        assert!(!session.is_open().unwrap());
        assert!(session.info().model_names().is_empty());
        session.close().unwrap();
        peer.run(&[]).unwrap();
        peer.close().unwrap();
        assert!(owner.upgrade().is_none());
    }
}
use lifecycle::{Error, SessionHandle};
pub mod dlpack {
    pub type BorrowedTensor<'a> = &'a [u8];
}
pub type RunResult = ();
pub struct SessionOptions(Arc<()>);
pub struct Session {
    owner: Option<Arc<()>>,
    wait: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
}
#[derive(Debug)]
pub struct RuntimeError(std::io::Error);
impl RuntimeError {
    pub fn code(&self) -> ErrorCode {
        ErrorCode::InvalidArgument
    }
    pub fn phase(&self) -> &str {
        "execute"
    }
}
impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}
impl Session {
    pub fn open(_: &Bundle, options: SessionOptions) -> Result<Self, RuntimeError> {
        Ok(Self {
            owner: Some(options.0),
            wait: None,
        })
    }
    pub fn is_open(&self) -> bool {
        self.owner.is_some()
    }
    pub fn new_session(&self) -> Result<Self, RuntimeError> {
        self.reset_check()?;
        Ok(Self {
            owner: self.owner.clone(),
            wait: None,
        })
    }
    pub fn close(&mut self) {
        self.owner.take();
    }
    fn reset_check(&self) -> Result<(), RuntimeError> {
        if self.is_open() {
            Ok(())
        } else {
            Err(RuntimeError(std::io::Error::other("session is closed")))
        }
    }
    pub fn reset(&mut self) -> Result<(), RuntimeError> {
        self.reset_check()
    }
    pub fn run(&mut self, _: &[(&str, dlpack::BorrowedTensor<'_>)]) -> Result<(), RuntimeError> {
        self.reset_check()?;
        if let Some((entered, resume)) = &self.wait {
            entered.send(()).unwrap();
            resume.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        Ok(())
    }
}
fn session() -> (SessionHandle, Weak<()>) {
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
        SessionHandle::open(&Bundle::open(path).unwrap(), SessionOptions(owner)).unwrap(),
        weak,
    )
}
#[test]
fn errors_preserve_the_source_chain_and_diagnostics() {
    let error = Error::Runtime(RuntimeError(std::io::Error::other("backend diagnostic")));
    assert!(error.to_string().contains("backend diagnostic"));
    assert_eq!(error.phase(), "execute");
    assert_eq!(error.code(), ErrorCode::InvalidArgument);
    assert!(std::error::Error::source(&error)
        .unwrap()
        .source()
        .is_some());
}
#[test]
fn close_waits_until_this_sessions_run_finishes() {
    let (session, owner) = session();
    let (entered, running) = mpsc::channel();
    let (resume, paused) = mpsc::channel();
    lifecycle::set_wait(&session, entered, paused);
    let session = Arc::new(session);
    let run = {
        let session = session.clone();
        thread::spawn(move || session.run(&[]).unwrap())
    };
    running.recv_timeout(Duration::from_secs(5)).unwrap();
    let (started, closing) = mpsc::channel();
    let (finished, closed) = mpsc::channel();
    let close = {
        let session = session.clone();
        thread::spawn(move || {
            started.send(()).unwrap();
            session.close().unwrap();
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
}
