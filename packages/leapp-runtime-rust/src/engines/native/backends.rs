// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native backend registration and the per-workspace execution contract.
//!
//! See `backends/README.md` for the contribution workflow and C ABI.

mod bridge;
mod nativert;
mod onnx;
mod torchscript;

use super::Transfers;
use crate::{
    bundle::schema::NodeSpec,
    dlpack::{BorrowedTensor, SharedTensor},
    ErrorCode, MemoryLocation, RuntimeError,
};
use anyhow::Result;

/// Backend plugin instance for one model node in one native workspace.
///
/// The workspace owns buffers/routing and calls this object exclusively, possibly
/// on different host threads (`Send`, not `Sync`). Peers share loaded models and
/// weights. Backends own their bindings, SDK handles and shared model lifetime.
/// Inputs are read-only: isolate SDK input mutation/retention inside the backend.
/// Complete all work accessing borrowed storage before returning, including
/// errors and unwinding. Drop bindings before buffers and retain libraries until
/// their final SDK handle is destroyed. Preparation failures must also be safe
/// to drop; the engine discards that workspace rather than silently falling back.
pub(super) trait BackendPlugin: Send {
    /// Create an unprepared peer sharing loaded models, weights and model-owned
    /// state. Execution-context state is private and starts fresh in each peer.
    /// Never reload artifacts or copy live bindings/intermediates.
    /// The engine allocates the peer's buffers and prepares its bindings later.
    /// Peer creation may overlap another peer's execution; read only immutable
    /// loaded metadata or synchronize shared access inside the backend.
    fn new_session(&self) -> Result<Box<dyn BackendPlugin>>;

    /// Whether peers need the engine's shared graph guard through output copying.
    /// This decision must remain stable across preparation and all related peers.
    /// Return false only when shared-model execution is safe across workspaces.
    /// The guard protects shared model state; execution-context state is private.
    fn needs_guard(&self) -> bool {
        true
    }

    /// Whether this loaded model permits fixed bindings. The engine additionally
    /// requires concrete input/output buffers. True permits an attempt, not a
    /// guarantee of success. False uses dynamic calls even for concrete shapes.
    fn can_prepare(&self) -> bool {
        false
    }

    /// Whether this plugin instance has successfully prepared valid bindings.
    /// New peers return false. The engine uses this to select the execution path.
    fn is_prepared(&self) -> bool {
        false
    }

    /// Bind buffers and cache tensor wrappers, metadata and argument arrays once.
    /// On success `is_prepared()` must return true. Contents can change between
    /// runs; addresses and layouts remain fixed. Copy/own retained metadata;
    /// argument slices and descriptor objects themselves are temporary.
    ///
    /// # Safety
    /// The caller supplies valid, ready input storage and writable output storage
    /// with matching dtype, shape and device. Inputs may alias each other; outputs
    /// must be disjoint from inputs and each other. Keep storage alive at fixed
    /// addresses until this plugin instance drops, including after failed preparation.
    /// During backend operations no other user may access that storage; between
    /// completed calls the engine may copy inputs/read outputs. The backend must
    /// finish all accesses before returning and may not mutate input storage.
    unsafe fn prepare(
        &mut self,
        _inputs: &[BorrowedTensor<'_>],
        _outputs: &[BorrowedTensor<'_>],
    ) -> Result<()> {
        Err(crate::error::Unsupported("backend does not support fixed bindings".into()).into())
    }

    /// Execute using successful preparation and current bound input contents.
    /// Write ready results into bound outputs, reusing bindings/metadata. Calling
    /// without preparation must return an error. No public snapshots are created
    /// here; the engine copies completed outputs before releasing its guard.
    fn run(&mut self) -> Result<()> {
        Err(crate::error::Unsupported("backend has no prepared execution path".into()).into())
    }

    /// Execute without fixed bindings, including concrete models that cannot be
    /// prepared. Inputs and the copy helper are borrowed only for this call.
    /// Return ready, independently owned outputs in declared port order; they
    /// must survive later runs/plugin destruction and must not alias inputs or
    /// mutable backend state. Use workspace transfers to isolate SDK-owned data.
    /// The engine validates output count/layout and routes results into its slots.
    fn run_dynamic(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        transfers: &mut Transfers,
    ) -> Result<Vec<SharedTensor>>;

    // Ownership diagnostics are test-only, not part of plugin execution/ABI.
    #[cfg(test)]
    fn loaded_identity(&self) -> usize;

    #[cfg(test)]
    fn prepared_identity(&self) -> Option<Vec<usize>> {
        None
    }
}

struct Backend {
    name: &'static str,
    validate: fn(&NodeSpec, MemoryLocation) -> Result<()>,
    load: fn(&NodeSpec, MemoryLocation) -> Result<Box<dyn BackendPlugin>>,
}

// One registration entry per canonical bundle backend. Lookup is load-time only.
const BACKENDS: &[Backend] = &[
    Backend {
        name: "onnx",
        validate: |_, _| Ok(()),
        load: |spec, location| Ok(Box::new(onnx::Model::load(spec, location)?)),
    },
    Backend {
        name: "jit",
        validate: |_, _| Ok(()),
        load: |spec, location| Ok(Box::new(torchscript::load(spec, location)?)),
    },
    Backend {
        name: "pt2",
        validate: |_, _| Ok(()),
        load: |spec, location| Ok(Box::new(nativert::load(spec, location)?)),
    },
];

fn backend(spec: &NodeSpec) -> Result<&'static Backend> {
    BACKENDS
        .iter()
        .find(|entry| entry.name == spec.parameters.backend)
        .ok_or_else(|| {
            crate::error::Unsupported(format!(
                "unsupported native backend {}; registered backends: {}",
                spec.parameters.backend,
                BACKENDS
                    .iter()
                    .map(|entry| entry.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .into()
        })
}

pub(super) fn validate(spec: &NodeSpec, location: MemoryLocation) -> Result<()> {
    (backend(spec)?.validate)(spec, location)
}

pub(super) fn load(spec: &NodeSpec, location: MemoryLocation) -> Result<Box<dyn BackendPlugin>> {
    let backend = backend(spec)?;
    (backend.validate)(spec, location)?;
    (backend.load)(spec, location)
}

pub(super) fn error(phase: &'static str, source: anyhow::Error) -> RuntimeError {
    use crate::engines::error::{Error, ErrorKind};
    let code = if let Some(error) = source.downcast_ref::<Error>() {
        match error.kind() {
            ErrorKind::InvalidArgument => ErrorCode::InvalidArgument,
            ErrorKind::NotFound => ErrorCode::NotFound,
            ErrorKind::Unsupported => ErrorCode::Unsupported,
            ErrorKind::Internal => ErrorCode::Internal,
        }
    } else if let Some(error) = source.downcast_ref::<ort::Error>() {
        match error.code() {
            ort::ErrorCode::InvalidArgument
            | ort::ErrorCode::InvalidGraph
            | ort::ErrorCode::InvalidProtobuf => ErrorCode::InvalidArgument,
            ort::ErrorCode::NoSuchFile => ErrorCode::NotFound,
            ort::ErrorCode::NotImplemented => ErrorCode::Unsupported,
            _ => ErrorCode::Internal,
        }
    } else {
        return RuntimeError::new(phase, source);
    };
    RuntimeError::classified(code, phase, source)
}
