// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private synchronous copies using reusable per-device transfer streams.

use crate::{
    dlpack::{BorrowedTensor, SharedTensor},
    engines::memory::{cuda::HandoffStream, dlpack, Buffer, OutputPool},
    MemoryLocation,
};
use anyhow::{ensure, Result};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub(super) struct Transfers {
    streams: HashMap<u32, HandoffStream>,
}

impl Transfers {
    pub fn new(location: MemoryLocation) -> Result<Self> {
        let mut value = Self::default();
        if let MemoryLocation::Cuda { device_id } = location {
            value
                .streams
                .insert(device_id, HandoffStream::new(device_id)?);
        }
        Ok(value)
    }

    pub fn copy(
        &mut self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
    ) -> Result<SharedTensor> {
        let (shape, len) = dlpack::layout(input)?;
        let buffer = Arc::new(Buffer::allocate_for_copy(len, target)?);
        // SAFETY: newly allocated output is exclusive and disjoint from input.
        unsafe {
            self.write(input, &buffer)?;
        }
        Ok(dlpack::from_buffer(
            input.descriptor().dtype,
            shape.to_vec(),
            buffer,
        )?)
    }

    pub fn copy_pooled(
        &mut self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
        pool: &OutputPool,
    ) -> Result<SharedTensor> {
        // SAFETY: write fills the reserved buffer and waits on the private stream,
        // including error paths, before the pool publishes or caches the result.
        unsafe { pool.copy_with(input, target, |input, buffer| self.write(input, buffer)) }
    }

    /// # Safety
    /// The buffer must be private, writable and disjoint from the ready input.
    pub unsafe fn write(&mut self, input: BorrowedTensor<'_>, buffer: &Buffer) -> Result<()> {
        let (_, len) = dlpack::layout(input)?;
        ensure!(len == buffer.len, "fixed buffer byte size mismatch");
        if len == 0 {
            return Ok(());
        }
        let descriptor = input.descriptor();
        let source = dlpack::location(descriptor.device)?;
        let device = match (source, buffer.location) {
            (_, MemoryLocation::Cuda { device_id }) | (MemoryLocation::Cuda { device_id }, _) => {
                Some(device_id)
            }
            _ => None,
        };
        let src = descriptor
            .data
            .cast::<u8>()
            .wrapping_add(descriptor.byte_offset as usize)
            .cast();
        // SAFETY: the caller owns the destination exclusively until the copy completes.
        unsafe {
            if let Some(device) = device {
                if let std::collections::hash_map::Entry::Vacant(entry) = self.streams.entry(device)
                {
                    entry.insert(HandoffStream::new(device)?);
                }
                self.streams[&device].copy_ready(
                    buffer.ptr.as_ptr().cast(),
                    buffer.location,
                    src,
                    source,
                    len,
                )?;
            } else {
                buffer.copy_from(src)?;
            }
        }
        Ok(())
    }
}
