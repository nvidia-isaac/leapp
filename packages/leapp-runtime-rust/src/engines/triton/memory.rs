// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(super) mod cuda;
pub(super) mod dlpack;

use crate::engines::triton::{native::sys, BorrowedTensor, Error, ErrorKind, Result, SharedTensor};
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
    pub(crate) fn native(self) -> (u32, i64) {
        match self {
            Self::Cpu => (sys::TRITONSERVER_MEMORY_CPU, 0),
            Self::PinnedCpu => (sys::TRITONSERVER_MEMORY_CPU_PINNED, 0),
            Self::Cuda { device_id } => (sys::TRITONSERVER_MEMORY_GPU, device_id.into()),
        }
    }

    pub(crate) fn validate(self) -> Result<()> {
        if self == Self::Cpu {
            return Ok(());
        }

        crate::engines::triton::memory::cuda::validate(self)
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
            crate::engines::triton::memory::cuda::allocate(len, location, zeroed)?
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
                crate::engines::triton::memory::cuda::copy(
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
                crate::engines::triton::memory::cuda::copy(
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
                crate::engines::triton::memory::cuda::free(self.ptr, self.location);
            }
        }
    }
}

// Bound cached references and bytes across all requests on this runtime. The
// key is raw byte length and location, independent of shape, dtype, or model.
// Overflow allocations remain ordinary owned outputs.
const MAX_ENTRIES: usize = 64;
const MAX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Default)]
pub(crate) struct OutputPool {
    entries: Mutex<Vec<Arc<Buffer>>>,
}

impl OutputPool {
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
        let mut fresh = None;
        let result = dlpack::copy_with(input, Some(target), |len, location| {
            if let Some(buffer) = self.reserve(len, location) {
                return Ok(buffer);
            }
            let buffer = Arc::new(Buffer::allocate_for_copy(len, location)?);
            fresh = Some(buffer.clone());
            Ok(buffer)
        })?;
        // A failed copy must never leave uninitialized storage in the pool.
        if let Some(buffer) = fresh {
            self.cache(buffer);
        }
        Ok(result)
    }

    fn reserve(&self, len: usize, location: MemoryLocation) -> Option<Arc<Buffer>> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
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
        if len == 0 || len > MAX_BYTES {
            return buffer;
        }
        let mut retired = Vec::new();
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let mut bytes: usize = entries.iter().map(|entry| entry.len).sum();
        // Evict only when necessary. Dropping a cached Arc never invalidates an
        // output owner; in-use entries are left alone to avoid allocator churn.
        let mut index = 0;
        while index < entries.len() && (entries.len() >= MAX_ENTRIES || bytes + len > MAX_BYTES) {
            if Arc::get_mut(&mut entries[index]).is_some() {
                let entry = entries.swap_remove(index);
                bytes -= entry.len;
                retired.push(entry);
            } else {
                index += 1;
            }
        }
        if entries.len() < MAX_ENTRIES && bytes + len <= MAX_BYTES {
            entries.push(Arc::clone(&buffer));
        }
        drop(entries);
        // cudaFree can wait for device work, including foreign reads of exposed
        // buffers. Never hold the pool mutex while destroying those allocations.
        drop(retired);
        buffer
    }
}

#[cfg(test)]
#[path = "../../../tests/unit_tests/triton/output_pool.rs"]
mod tests;
