// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal execution contract and shared tensor storage.
pub(crate) mod error;
pub(crate) mod memory;
pub(crate) mod triton;

use crate::{
    dlpack::{BorrowedTensor, SharedTensor},
    OutputPlacement, RuntimeError,
};
use indexmap::IndexMap;
use memory::OutputPool;
use std::sync::Arc;

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

/// A loaded engine is shared by independent sessions. Calls complete before returning.
pub(crate) trait Engine: Send + Sync {
    fn run(
        &self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        placement: &OutputPlacement,
        pool: &Arc<OutputPool>,
    ) -> Result<IndexMap<String, SharedTensor>, RuntimeError>;
    /// Always produce independent, ready storage, even at the same location.
    fn copy(
        &self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
        pool: Option<&OutputPool>,
    ) -> Result<SharedTensor, RuntimeError> {
        match pool {
            Some(pool) => pool.copy(input, target),
            None => memory::dlpack::copy(input, Some(target)),
        }
        .map_err(Into::into)
    }
}

pub(crate) fn open(
    bundle: &crate::Bundle,
    graph: &crate::session::graph::Graph<'_>,
    options: &crate::SessionOptions,
) -> Result<Arc<dyn Engine>, RuntimeError> {
    triton::open(bundle, graph, options)
}

/// DLPack producer handoff support for language bindings.
pub(crate) use memory::cuda::HandoffStream;
