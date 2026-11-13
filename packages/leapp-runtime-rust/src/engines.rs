// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime engine selection and the internal execution contract.
pub(crate) mod error;
pub(crate) mod memory;
mod native;
pub(crate) mod triton;

use crate::{
    dlpack::{BorrowedTensor, OutputBuffer, SharedTensor},
    OutputPlacement, RuntimeError,
};
use indexmap::IndexMap;
use std::sync::Arc;

/// Execution engine selected when opening a session.
/// Defaults to native.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EngineKind {
    Triton,
    #[default]
    Native,
}

/// Device used by model instances, independently of input and output storage.
#[derive(Debug, Clone, Copy, Default)]
pub enum ExecutionDevice {
    #[default]
    Cpu,
    Cuda {
        device_id: u32,
    },
}

/// Storage location, independent of a model's execution device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MemoryLocation {
    #[default]
    Cpu,
    PinnedCpu,
    Cuda {
        device_id: u32,
    },
}

/// Session execution over shared loaded models. Calls complete before returning.
/// Session output destinations are validated at construction; internal callers
/// using other placements validate those before execution. Native uses the input
/// order established by Session: public declarations, then private defaults and
/// feedback in model declaration order. Session checks names and completeness,
/// plus supplied output names, declared dtype/shape, and output/output overlap.
/// Engines validate input/output overlap before execution and never retain buffers.
pub(crate) trait Engine: Send + Sync {
    fn new_session(self: Arc<Self>) -> Result<Arc<dyn Engine>, RuntimeError>;

    fn run(
        &self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        placement: &OutputPlacement,
        outputs: &mut std::collections::HashMap<&str, OutputBuffer<'_>>,
    ) -> Result<IndexMap<String, SharedTensor>, RuntimeError>;
    /// Always produce independent, ready storage, even at the same location.
    fn copy(
        &self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
    ) -> Result<SharedTensor, RuntimeError> {
        memory::dlpack::copy(input, Some(target)).map_err(Into::into)
    }
}

pub(crate) fn open(
    bundle: &crate::Bundle,
    graph: &crate::session::graph::Graph<'_>,
    options: &crate::SessionOptions,
) -> Result<Arc<dyn Engine>, RuntimeError> {
    match options.engine {
        EngineKind::Triton => triton::open(bundle, graph, options),
        EngineKind::Native => native::open(bundle, graph, options),
    }
}

/// DLPack producer handoff support for language bindings.
pub(crate) use memory::cuda::HandoffStream;

#[cfg(test)]
#[path = "../tests/functional_tests/session.rs"]
pub(crate) mod tests;
