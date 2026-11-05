// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod cuda;
pub(crate) mod dlpack;

use crate::dlpack::{BorrowedTensor, SharedTensor};
use crate::engines::error::{Error, ErrorKind, Result};
use crate::MemoryLocation;
use std::{
    alloc::{alloc, alloc_zeroed, dealloc, Layout},
    ffi::c_void,
    mem::align_of,
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

impl MemoryLocation {
    pub(crate) fn validate(self) -> Result<()> {
        if self == Self::Cpu {
            return Ok(());
        }

        crate::engines::memory::cuda::validate(self)
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidArgument, "buffer", message)
}

pub(crate) struct Buffer {
    pub(crate) ptr: NonNull<u8>,
    pub(crate) len: usize,
    pub(crate) location: MemoryLocation,
    exposed: AtomicBool,
}

// SAFETY: allocation metadata is immutable; freeing on any thread selects the
// correct CUDA device. Data access occurs only through explicit borrowed contracts.
unsafe impl Send for Buffer {}

unsafe impl Sync for Buffer {}

impl Buffer {
    pub(crate) fn allocate(len: usize, location: MemoryLocation) -> Result<Self> {
        Self::allocate_storage(len, location, true)
    }

    // The caller must fill all bytes before publishing or caching this buffer.
    pub(crate) fn allocate_for_copy(len: usize, location: MemoryLocation) -> Result<Self> {
        Self::allocate_storage(len, location, false)
    }

    fn allocate_storage(len: usize, location: MemoryLocation, zeroed: bool) -> Result<Self> {
        location.validate()?;
        let layout = Layout::from_size_align(len, align_of::<u64>())
            .map_err(|_| invalid("invalid allocation size"))?;
        let ptr = if len == 0 {
            NonNull::<u64>::dangling().cast()
        } else if location == MemoryLocation::Cpu {
            // SAFETY: layout is nonzero and valid. Backend output allocations are
            // zeroed; copy destinations are completely filled before publication.
            NonNull::new(unsafe {
                if zeroed {
                    alloc_zeroed(layout)
                } else {
                    alloc(layout)
                }
            })
            .ok_or_else(|| Error::new(ErrorKind::Internal, "buffer", "CPU allocation failed"))?
        } else {
            crate::engines::memory::cuda::allocate(len, location, zeroed)?
        };
        Ok(Self {
            ptr,
            len,
            location,
            exposed: AtomicBool::new(false),
        })
    }

    // A foreign consumer may queue device work beyond the lifetime of its lease.
    // Without a completion event, exposed allocations never return to the pool.
    pub(crate) fn expose(&self) {
        self.exposed.store(true, Ordering::Release);
    }

    pub(crate) fn can_reuse(&self) -> bool {
        !self.exposed.load(Ordering::Acquire)
    }

    pub(crate) unsafe fn copy_from(&self, source: *const c_void) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }

        if let MemoryLocation::Cuda { device_id } = self.location {
            // SAFETY: caller provides a readable source; self owns the destination.
            return unsafe {
                crate::engines::memory::cuda::copy(
                    self.ptr.as_ptr().cast(),
                    source,
                    self.len,
                    device_id,
                    true,
                )
            };
        }
        // SAFETY: caller provides len readable bytes, disjoint from this allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(source.cast::<u8>(), self.ptr.as_ptr(), self.len);
        }
        Ok(())
    }

    pub(crate) unsafe fn copy_to(&self, destination: *mut c_void) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }

        if let MemoryLocation::Cuda { device_id } = self.location {
            // SAFETY: caller provides a writable destination; self owns the source.
            return unsafe {
                crate::engines::memory::cuda::copy(
                    destination,
                    self.ptr.as_ptr().cast(),
                    self.len,
                    device_id,
                    false,
                )
            };
        }
        // SAFETY: caller provides len writable bytes, disjoint from this allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), destination.cast::<u8>(), self.len);
        }
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        if self.location == MemoryLocation::Cpu {
            // SAFETY: the layout matches allocation and this is its final owner.
            unsafe {
                dealloc(
                    self.ptr.as_ptr(),
                    Layout::from_size_align_unchecked(self.len, align_of::<u64>()),
                );
            }
        } else {
            // SAFETY: this is the final owner of the CUDA allocation.
            unsafe {
                crate::engines::memory::cuda::free(self.ptr, self.location);
            }
        }
    }
}

// Retain a configurable number of buffers. The bundle supplies the largest
// static output size; actual dynamic outputs can raise that per-entry budget.
// This bounds retained bytes without excluding a valid large tensor from reuse.
pub(crate) struct OutputPool {
    options: crate::OutputPoolOptions,
    initial_bytes: usize,
    state: Mutex<PoolState>,
}

struct PoolState {
    entries: Vec<Arc<Buffer>>,
    largest_buffer: usize,
}

impl OutputPool {
    pub(crate) fn new(options: crate::OutputPoolOptions, largest_buffer: usize) -> Self {
        Self {
            options,
            initial_bytes: largest_buffer,
            state: Mutex::new(PoolState {
                entries: Vec::new(),
                largest_buffer,
            }),
        }
    }

    pub(crate) fn fresh(&self) -> Self {
        Self::new(self.options, self.initial_bytes)
    }

    pub(crate) fn clear(&self) {
        let retired =
            std::mem::take(&mut self.state.lock().unwrap_or_else(|e| e.into_inner()).entries);
        // CUDA destruction may synchronize. Never free while holding the lock.
        drop(retired);
    }

    pub(crate) fn allocate(&self, len: usize, location: MemoryLocation) -> Result<Arc<Buffer>> {
        if let Some(buffer) = self.reserve(len, location) {
            return Ok(buffer);
        }
        // Allocation may synchronize CUDA. Do not hold the pool mutex across it.
        Ok(self.cache(Arc::new(Buffer::allocate(len, location)?)))
    }

    pub(crate) fn copy(
        &self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
    ) -> Result<SharedTensor> {
        // SAFETY: write_buffer fills the exclusive destination synchronously.
        unsafe {
            self.copy_with(input, target, |input, buffer| {
                dlpack::write_buffer(input, buffer)
            })
        }
    }

    /// Copy into a reserved buffer using the engine's existing transfer mechanism.
    ///
    /// # Safety
    /// `copy` must initialize every destination byte before returning success and
    /// finish all accesses before returning, including on error or unwinding.
    /// The destination is exclusive and disjoint from the ready input.
    pub(crate) unsafe fn copy_with<E: From<Error>>(
        &self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
        copy: impl FnOnce(BorrowedTensor<'_>, &Buffer) -> std::result::Result<(), E>,
    ) -> std::result::Result<SharedTensor, E> {
        let (shape, len) = dlpack::layout(input)?;
        let (buffer, fresh) = match self.reserve(len, target) {
            Some(buffer) => (buffer, false),
            None => (Arc::new(Buffer::allocate_for_copy(len, target)?), true),
        };
        copy(input, &buffer)?;
        let fresh = fresh.then(|| buffer.clone());
        let result = dlpack::from_buffer(input.descriptor().dtype, shape.to_vec(), buffer)?;
        // A failed copy must never leave uninitialized storage in the pool.
        if let Some(buffer) = fresh {
            self.cache(buffer);
        }
        Ok(result)
    }

    fn reserve(&self, len: usize, location: MemoryLocation) -> Option<Arc<Buffer>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let entries = &mut state.entries;
        for entry in entries.iter_mut() {
            if entry.len == len
                && entry.location == location
                && Arc::get_mut(entry).is_some_and(|buffer| buffer.can_reuse())
            {
                // Reserve under the lock; exclude both retained and weak owners.
                return Some(Arc::clone(entry));
            }
        }
        None
    }

    fn cache(&self, buffer: Arc<Buffer>) -> Arc<Buffer> {
        let len = buffer.len;
        if len == 0 || self.options.max_entries == 0 {
            return buffer;
        }
        let mut retired = Vec::new();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.largest_buffer = state.largest_buffer.max(len);
        let max_bytes = state
            .largest_buffer
            .saturating_mul(self.options.max_entries);
        let entries = &mut state.entries;
        let mut bytes: usize = entries.iter().map(|entry| entry.len).sum();
        // Keep retained/weak owners safe. Drop idle references only as needed;
        // overflow allocations still succeed without becoming cache entries.
        let mut index = 0;
        while index < entries.len()
            && (entries.len() >= self.options.max_entries || len > max_bytes.saturating_sub(bytes))
        {
            if Arc::get_mut(&mut entries[index]).is_some() {
                let entry = entries.swap_remove(index);
                bytes -= entry.len;
                retired.push(entry);
            } else {
                index += 1;
            }
        }
        if entries.len() < self.options.max_entries && len <= max_bytes.saturating_sub(bytes) {
            entries.push(Arc::clone(&buffer));
        }
        drop(state);
        drop(retired);
        buffer
    }
}

#[cfg(test)]
#[path = "../../tests/unit_tests/memory/output_pool.rs"]
mod tests;
