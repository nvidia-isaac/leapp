// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native backend registration and the per-workspace execution contract.
//!
//! See `backends/README.md` for the Rust/C++ contribution workflow.

mod bridge;
mod onnx;

use super::Transfers;
use crate::{
    bundle::schema::NodeSpec,
    dlpack::{BorrowedTensor, DLTensor, SharedTensor},
    ErrorCode, MemoryLocation, RuntimeError,
};
use anyhow::Result;

/// A prepared argument replaced by an invocation-scoped Session input.
/// Keep this layout paired with the private C++ adapter ABI.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct InputRoute {
    pub position: usize,
    pub boundary: usize,
}

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

    /// Required alignment for borrowed prepared inputs, queried at initialization.
    /// Zero disables borrowing. Placement/alignment are resolved before execution.
    fn input_alignment(&self) -> usize {
        0
    }

    /// Bind buffers and cache tensor wrappers, metadata and argument arrays once.
    /// On success `is_prepared()` must return true. Contents can change between
    /// runs; layouts remain fixed. Copy/own retained metadata;
    /// argument slices and descriptor objects themselves are temporary.
    ///
    /// # Safety
    /// The caller supplies valid, ready input storage and writable output storage
    /// with matching dtype, shape and device. Inputs may alias each other; outputs
    /// must be disjoint from inputs and each other. Keep storage alive at fixed
    /// addresses until this plugin instance drops, including failed preparation.
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

    /// Required output address alignment, stable across all invocations.
    fn output_alignment(&self) -> usize {
        1
    }

    /// Execute once with prepared bindings and selected boundary inputs.
    /// Each route maps an argument position to its index in Session's inputs.
    /// Unrouted arguments retain their prepared workspace bindings. The engine
    /// has already established input compatibility, placement and alignment.
    /// Complete all work and restore bindings before returning, including errors.
    /// Caller pointers must never be retained. Execution errors propagate without retry.
    /// Empty routes reuse all prepared inputs. There is one prepared execution
    /// entry point, with no borrowing/staging selection during dispatch.
    /// # Safety
    /// Output bindings are writable, disjoint, compatible, and alive through completion.
    unsafe fn run(
        &mut self,
        _inputs: &[(&str, BorrowedTensor<'_>)],
        _routes: &[InputRoute],
        _outputs: &[(usize, DLTensor)],
    ) -> Result<()> {
        Err(crate::error::Unsupported("backend has no prepared execution path".into()).into())
    }

    /// Execute without fixed bindings, including concrete models that cannot be
    /// prepared. Inputs and the copy helper are borrowed only for this call.
    /// Return one entry per output: None for supplied targets, Some for owned results.
    /// Return valid, contiguous, ready, independently owned outputs on the
    /// workspace execution device, matching the declared output count, order,
    /// dtype, rank and fixed dimensions. SDK metadata establishes these guarantees
    /// at initialization or the backend checks unresolved values before return. They
    /// must survive later runs/plugin destruction and must not alias inputs or
    /// mutable backend state. Use workspace transfers to isolate SDK-owned data.
    /// Inputs have already passed layout, device and port validation. Bundle
    /// validation requires identical declarations across connected ports.
    /// # Safety
    /// Supplied outputs satisfy the same contract as run and are not retained.
    unsafe fn run_dynamic(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        outputs: &[(usize, DLTensor)],
        transfers: &mut Transfers,
    ) -> Result<Vec<Option<SharedTensor>>>;

    #[cfg(test)]
    fn loaded_identity(&self) -> usize;

    #[cfg(test)]
    fn prepared_identity(&self) -> Option<Vec<usize>> {
        None
    }
}

type LoadBackend = fn(
    &NodeSpec,
    MemoryLocation,
    Option<crate::CudaStream>,
    Option<&std::collections::HashMap<String, String>>,
) -> Result<Box<dyn BackendPlugin>>;

struct Backend {
    name: &'static str,
    validate: fn(&NodeSpec, MemoryLocation) -> Result<()>,
    load: LoadBackend,
}

// One registration entry per canonical bundle backend. Lookup is load-time only.
const BACKENDS: &[Backend] = &[
    Backend {
        name: "onnx",
        validate: |_, _| Ok(()),
        load: |spec, location, stream, options| {
            Ok(Box::new(onnx::Model::load(
                spec, location, stream, options,
            )?))
        },
    },
    Backend {
        name: "jit",
        validate: |_, _| Ok(()),
        load: |spec, location, stream, _| {
            bridge::load(
                spec,
                location,
                stream,
                "torchscript",
                "libleapp_torchscript.so",
            )
        },
    },
    Backend {
        name: "pt2",
        validate: |_, _| Ok(()),
        load: |spec, location, stream, _| {
            bridge::load(
                spec,
                location,
                stream,
                "nativert",
                "libleapp_nativert_backend.so",
            )
        },
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

pub(super) fn load(
    spec: &NodeSpec,
    location: MemoryLocation,
    stream: Option<crate::CudaStream>,
    onnx_tensorrt: Option<&std::collections::HashMap<String, String>>,
) -> Result<Box<dyn BackendPlugin>> {
    let backend = backend(spec)?;
    (backend.validate)(spec, location)?;
    (backend.load)(spec, location, stream, onnx_tensorrt)
}

pub(super) fn error(phase: &'static str, source: anyhow::Error) -> RuntimeError {
    use crate::engines::error::{Error, ErrorKind};
    let code = if let Some(error) = source.downcast_ref::<Error>() {
        match error.kind() {
            ErrorKind::InvalidArgument => ErrorCode::InvalidArgument,
            ErrorKind::NotFound => ErrorCode::NotFound,
            ErrorKind::Unsupported => ErrorCode::Unsupported,
            ErrorKind::Io => ErrorCode::Io,
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
