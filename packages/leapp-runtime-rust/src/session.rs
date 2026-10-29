// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public Session API. Execution is supplied by the runtime integration.

use crate::{
    dlpack::{BorrowedTensor, SharedTensor},
    Bundle, ExecutionDevice, MemoryLocation, RuntimeError,
};
use indexmap::IndexMap;
use std::collections::HashMap;

/// Exact allocation locations keyed by logical output names.
#[derive(Debug, Clone, Default)]
pub struct OutputPlacement {
    pub default_location: MemoryLocation,
    pub overrides: HashMap<String, MemoryLocation>,
}

/// Retention policy for each session's reusable output buffers.
///
/// The runtime derives the byte budget from bundle tensor shapes, dtypes, and
/// placement, growing it for actual dynamic output sizes. There is no fixed
/// byte limit that excludes large tensors from reuse. This policy does not cap
/// memory held by models, fixed workspaces, or retained results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputPoolOptions {
    /// Maximum number of cached buffers. Zero disables caching, not allocation.
    pub max_entries: usize,
}

impl Default for OutputPoolOptions {
    fn default() -> Self {
        Self { max_entries: 64 }
    }
}

/// Immutable construction options, inherited by every new session.
///
/// ```
/// use lrt::{OutputPoolOptions, SessionOptions};
///
/// let options = SessionOptions {
///     output_pool: OutputPoolOptions { max_entries: 0 },
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct SessionOptions {
    pub device: ExecutionDevice,
    pub output_placement: OutputPlacement,
    /// Each session has its own cache; peers inherit this policy, not its buffers.
    pub output_pool: OutputPoolOptions,
}

/// Managed DLPack public outputs in bundle declaration order, independent of the session.
/// Repeated output declarations share one named entry. Cloning shares storage.
#[derive(Clone)]
pub struct RunResult {
    outputs: IndexMap<String, SharedTensor>,
}

impl RunResult {
    /// Look up an output by its qualified LEAPP `node/port` name.
    pub fn get(&self, name: &str) -> Option<&SharedTensor> {
        self.outputs.get(name)
    }

    /// Iterate through public names and tensors in declaration order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &SharedTensor)> {
        self.outputs
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }
}

/// Independent feedback state sharing loaded models with sessions from `new_session`.
/// Separate sessions can execute concurrently. Model-internal state remains shared.
///
/// This API revision defines the contract but does not implement execution.
/// [`Session::open`] returns [`crate::ErrorCode::Unsupported`].
pub struct Session {
    bundle: Bundle,
}

impl Session {
    /// Load models and capture initial values on the execution device.
    /// Dynamic state requires an exported initial value; other absent values use zeros.
    /// Execution is unavailable in this API-only revision.
    pub fn open(_bundle: &Bundle, _options: SessionOptions) -> Result<Self, RuntimeError> {
        Err(execution_unavailable("load"))
    }

    /// Immutable bundle snapshot, also available after close.
    pub fn info(&self) -> &Bundle {
        &self.bundle
    }

    /// Whether this session retains its loaded execution engine.
    pub fn is_open(&self) -> bool {
        false
    }

    /// Create an independently initialized session without reloading models or files.
    /// Configuration and captured initial values are inherited; current feedback is not.
    /// The peer starts with an empty output cache using the same retention policy.
    pub fn new_session(&self) -> Result<Self, RuntimeError> {
        Err(execution_unavailable("load"))
    }

    /// Release this session's state and engine reference. Other sessions and results survive.
    pub fn close(&mut self) {}

    /// Restore captured state without rereading files or resetting model-internal state.
    pub fn reset(&mut self) -> Result<(), RuntimeError> {
        Err(execution_unavailable("state"))
    }

    /// Borrow every public input exactly once until synchronous execution completes.
    /// Names and tensors are borrowed from the map; matching is by name, independent
    /// of iteration order. Missing or unknown inputs are errors. Inserting a duplicate
    /// key replaces its earlier value before the runtime receives the map.
    /// Feedback advances only after every output and isolation copy succeeds.
    ///
    /// ```
    /// use lrt::{dlpack::BorrowedTensor, RunResult, RuntimeError, Session};
    /// use std::collections::HashMap;
    ///
    /// fn step(session: &mut Session) -> Result<RunResult, RuntimeError> {
    ///     let values = [1_f32, 2., 3.];
    ///     let name = String::from("model/input");
    ///     let inputs = HashMap::from([
    ///         (name.as_str(), BorrowedTensor::cpu(&values, &[3]).unwrap()),
    ///     ]);
    ///     session.run(&inputs)
    /// }
    /// ```
    pub fn run(
        &mut self,
        _inputs: &HashMap<&str, BorrowedTensor<'_>>,
    ) -> Result<RunResult, RuntimeError> {
        Err(execution_unavailable("execute"))
    }
}

fn execution_unavailable(phase: &'static str) -> RuntimeError {
    RuntimeError::new(
        phase,
        crate::error::Unsupported(
            "Session execution is not implemented in this API revision".into(),
        )
        .into(),
    )
}
