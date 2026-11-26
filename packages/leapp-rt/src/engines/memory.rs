// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod cuda;
pub(crate) mod dlpack;

use crate::engines::error::{Error, ErrorKind, Result};
use crate::MemoryLocation;
use std::{
    alloc::{alloc, alloc_zeroed, dealloc, Layout},
    ffi::c_void,
    mem::align_of,
    ptr::NonNull,
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
}

// SAFETY: allocation metadata is immutable; freeing on any thread selects the
// correct CUDA device. Data access occurs only through explicit borrowed contracts.
unsafe impl Send for Buffer {}

unsafe impl Sync for Buffer {}

impl Buffer {
    pub(crate) fn allocate(len: usize, location: MemoryLocation) -> Result<Self> {
        location.validate()?;
        Self::allocate_validated(len, location, true)
    }

    // The caller must fill all bytes before publishing this buffer.
    pub(crate) fn allocate_for_copy(len: usize, location: MemoryLocation) -> Result<Self> {
        location.validate()?;
        Self::allocate_validated(len, location, false)
    }

    // Location must already be validated. Fill uninitialized storage before publication.
    pub(crate) fn allocate_validated(
        len: usize,
        location: MemoryLocation,
        zeroed: bool,
    ) -> Result<Self> {
        let layout = Layout::from_size_align(len, align_of::<u64>())
            .map_err(|_| invalid("invalid allocation size"))?;
        let ptr = if len == 0 {
            NonNull::<u64>::dangling().cast()
        } else if location == MemoryLocation::Cpu {
            // SAFETY: layout is nonzero and valid. Uninitialized destinations
            // are completely filled before publication.
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
        Ok(Self { ptr, len, location })
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
