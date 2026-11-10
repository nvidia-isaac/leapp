// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synchronous copies using reusable per-device transfer streams.

use crate::{
    dlpack::{BorrowedTensor, SharedTensor},
    engines::memory::{cuda::HandoffStream, dlpack, Buffer},
    MemoryLocation,
};
use anyhow::Result;
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub(super) struct Transfers {
    streams: HashMap<u32, HandoffStream>,
}

impl Transfers {
    pub fn new(location: MemoryLocation, options: &crate::SessionOptions) -> Result<Self> {
        let mut value = Self {
            streams: options
                .cuda_streams
                .iter()
                .map(|(&device, &stream)| (device, HandoffStream::borrowed(device, stream)))
                .collect(),
        };
        for location in std::iter::once(location)
            .chain(std::iter::once(options.output_placement.default_location))
            .chain(options.output_placement.overrides.values().copied())
        {
            if let MemoryLocation::Cuda { device_id } = location {
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    value.streams.entry(device_id)
                {
                    entry.insert(HandoffStream::new(device_id)?);
                }
            }
        }
        Ok(value)
    }

    pub fn copy(
        &mut self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
    ) -> Result<SharedTensor> {
        let (_, len) = dlpack::layout(input)?;
        // SAFETY: layout checked the metadata, location and byte count of this borrow.
        unsafe { self.copy_validated(input, target, len) }
    }

    /// # Safety
    /// Input must have a validated, unchanged contiguous layout of len bytes.
    pub unsafe fn copy_validated(
        &mut self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
        len: usize,
    ) -> Result<SharedTensor> {
        let descriptor = input.descriptor();
        let shape = if descriptor.ndim == 0 {
            &[]
        } else {
            // SAFETY: the caller validated the metadata, retained by the input borrow.
            unsafe { std::slice::from_raw_parts(descriptor.shape, descriptor.ndim as usize) }
        };
        let buffer = Arc::new(Buffer::allocate_for_copy(len, target)?);
        // SAFETY: newly allocated output is exclusive and disjoint from input.
        unsafe {
            self.write(input, &buffer)?;
        }
        // SAFETY: the input layout was checked and write initialized every byte.
        Ok(unsafe { dlpack::from_buffer(input.descriptor().dtype, shape.to_vec(), buffer)? })
    }

    /// # Safety
    /// The input layout must match the private, writable, disjoint buffer's byte size.
    pub unsafe fn write(&mut self, input: BorrowedTensor<'_>, buffer: &Buffer) -> Result<()> {
        let len = buffer.len;
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
                // Configured execution/output devices are initialized up front.
                // A staged input can introduce another CUDA device later.
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
