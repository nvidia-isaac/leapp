// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Error, ErrorKind, MemoryLocation, Result};
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    ptr::{self, NonNull},
};

// CUDA runtime ABI, limited to allocation, synchronous copies and device selection.
#[allow(non_snake_case)]
extern "C" {
    fn cudaGetErrorString(error: c_int) -> *const c_char;

    fn cudaGetDeviceCount(count: *mut c_int) -> c_int;

    fn cudaGetDevice(device: *mut c_int) -> c_int;

    fn cudaSetDevice(device: c_int) -> c_int;

    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;

    fn cudaFree(ptr: *mut c_void) -> c_int;

    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> c_int;

    fn cudaFreeHost(ptr: *mut c_void) -> c_int;

    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> c_int;

    fn cudaMemset(ptr: *mut c_void, value: c_int, count: usize) -> c_int;

    fn cudaDeviceSynchronize() -> c_int;

    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> c_int;

    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;

    fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
}

const HOST_ALLOC_PORTABLE: u32 = 1;
const COPY_HOST_TO_DEVICE: c_int = 1;
const COPY_DEVICE_TO_HOST: c_int = 2;
const COPY_DEVICE_TO_DEVICE: c_int = 3;
const STREAM_NON_BLOCKING: u32 = 1;

fn check(code: c_int) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: CUDA returns a static NUL-terminated error description.
    let message = unsafe { CStr::from_ptr(cudaGetErrorString(code)) }
        .to_string_lossy()
        .into_owned();
    Err(Error::new(ErrorKind::Internal, "buffer", message))
}

pub(crate) fn validate(location: MemoryLocation) -> Result<()> {
    if let MemoryLocation::Cuda { device_id } = location {
        let mut count = 0;
        // SAFETY: count is a live output parameter.
        unsafe {
            check(cudaGetDeviceCount(&mut count))?;
        }
        if i64::from(device_id) >= i64::from(count) {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "buffer",
                "CUDA device is unavailable",
            ));
        }
    }
    Ok(())
}

struct Device {
    previous: c_int,
    changed: bool,
}

impl Device {
    fn select(device: u32) -> Result<Self> {
        let device = c_int::try_from(device).map_err(|_| {
            Error::new(
                ErrorKind::InvalidArgument,
                "buffer",
                "CUDA device ID is out of range",
            )
        })?;
        let mut previous = 0;
        // SAFETY: selecting a device affects only the current thread; Drop restores it.
        unsafe {
            check(cudaGetDevice(&mut previous))?;
            if previous != device {
                check(cudaSetDevice(device))?;
            }
        }
        Ok(Self {
            previous,
            changed: previous != device,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if self.changed {
            // SAFETY: previous was returned by CUDA on this thread.
            unsafe {
                cudaSetDevice(self.previous);
            }
        }
    }
}

/// Wait for work on a stream owned by an external tensor producer.
///
/// # Safety
/// `address` must identify a live CUDA stream on `device` until this call returns.
/// The owning producer must not destroy the stream concurrently.
pub unsafe fn synchronize_external_stream(device: u32, address: usize) -> Result<()> {
    let _device = Device::select(device)?;
    // SAFETY: the caller retains a valid stream on the selected device.
    unsafe { check(cudaStreamSynchronize(address as *mut c_void)) }
}

pub(crate) fn allocate(len: usize, location: MemoryLocation) -> Result<NonNull<u8>> {
    let mut raw = ptr::null_mut();
    // SAFETY: raw is writable and the caller validated location and allocation size.
    unsafe {
        match location {
            MemoryLocation::PinnedCpu => {
                check(cudaHostAlloc(&mut raw, len, HOST_ALLOC_PORTABLE))?;
                ptr::write_bytes(raw.cast::<u8>(), 0, len);
            }
            MemoryLocation::Cuda { device_id } => {
                let _device = Device::select(device_id)?;
                check(cudaMalloc(&mut raw, len))?;
                if let Err(error) =
                    check(cudaMemset(raw, 0, len)).and_then(|_| check(cudaDeviceSynchronize()))
                {
                    cudaFree(raw);
                    return Err(error);
                }
            }
            MemoryLocation::Cpu => unreachable!(),
        }
    }
    NonNull::new(raw.cast())
        .ok_or_else(|| Error::new(ErrorKind::Internal, "buffer", "CUDA returned null storage"))
}

pub(crate) unsafe fn free(ptr: NonNull<u8>, location: MemoryLocation) {
    // SAFETY: caller transfers the final ownership of this CUDA allocation.
    unsafe {
        match location {
            MemoryLocation::PinnedCpu => {
                cudaFreeHost(ptr.as_ptr().cast());
            }
            MemoryLocation::Cuda { device_id } => {
                if let Ok(_device) = Device::select(device_id) {
                    cudaFree(ptr.as_ptr().cast());
                }
            }
            MemoryLocation::Cpu => unreachable!(),
        }
    }
}

pub(crate) unsafe fn copy(
    dst: *mut c_void,
    src: *const c_void,
    len: usize,
    device: u32,
    to_device: bool,
) -> Result<()> {
    let _device = Device::select(device)?;
    // SAFETY: caller supplies valid regions and the correct transfer direction.
    unsafe {
        check(cudaMemcpy(
            dst,
            src,
            len,
            if to_device {
                COPY_HOST_TO_DEVICE
            } else {
                COPY_DEVICE_TO_HOST
            },
        ))?;
        check(cudaDeviceSynchronize())
    }
}

/// A nonblocking CUDA stream used to receive producer completion dependencies.
pub struct HandoffStream {
    stream: *mut c_void,
    device: u32,
}

// SAFETY: the stream is owned, CUDA supports host-thread handoff, and every
// operation selects its device. Destruction requires exclusive ownership.
unsafe impl Send for HandoffStream {}

unsafe impl Sync for HandoffStream {}

impl HandoffStream {
    /// Create a stream on the selected device without changing the caller's device.
    pub fn new(device: u32) -> Result<Self> {
        let _device = Device::select(device)?;
        let mut stream = ptr::null_mut();
        // SAFETY: stream is a live output parameter on the selected device.
        unsafe {
            check(cudaStreamCreateWithFlags(&mut stream, STREAM_NON_BLOCKING))?;
        }
        Ok(Self { stream, device })
    }

    /// Invoke a producer on this stream's device, then restore the caller's device.
    /// The callback runs synchronously on the current host thread.
    pub fn with_device<T>(&self, producer: impl FnOnce() -> T) -> Result<T> {
        let mut device = Device::select(self.device)?;
        // A producer may change the current device even if select did not.
        device.changed = true;
        Ok(producer())
    }

    /// Handle passed to a DLPack producer's stream argument.
    pub fn address(&self) -> usize {
        self.stream as usize
    }

    /// Wait for producer work before handing data to Triton's internal streams.
    pub fn synchronize(&self) -> Result<()> {
        let _device = Device::select(self.device)?;
        // SAFETY: this object owns a live stream on the selected device.
        unsafe { check(cudaStreamSynchronize(self.stream)) }
    }
}

impl Drop for HandoffStream {
    fn drop(&mut self) {
        if let Ok(_device) = Device::select(self.device) {
            // SAFETY: this object is the final owner of the stream.
            unsafe {
                cudaStreamDestroy(self.stream);
            }
        }
    }
}

pub(crate) unsafe fn copy_device(
    dst: *mut c_void,
    src: *const c_void,
    len: usize,
    device: u32,
) -> Result<()> {
    let _device = Device::select(device)?;
    // SAFETY: caller supplies disjoint live CUDA allocations on this device.
    unsafe {
        check(cudaMemcpy(dst, src, len, COPY_DEVICE_TO_DEVICE))?;
        check(cudaDeviceSynchronize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn handoff_producer_restores_device_on_return_error_and_panic() {
        let (mut original, mut count) = (0, 0);
        // SAFETY: both arguments are live output parameters.
        unsafe {
            check(cudaGetDevice(&mut original)).unwrap();
            check(cudaGetDeviceCount(&mut count)).unwrap();
        }
        let target = ((original + 1) % count) as u32;
        let stream = HandoffStream::new(target).unwrap();
        let current = || {
            let mut device = -1;
            // SAFETY: device is a live output parameter.
            unsafe { check(cudaGetDevice(&mut device)).unwrap() };
            device
        };
        assert_eq!(stream.with_device(current).unwrap(), target as c_int);
        assert_eq!(current(), original);
        let error = stream
            .with_device(|| {
                assert_eq!(current(), target as c_int);
                Err::<(), _>("producer failed")
            })
            .unwrap();
        assert!(error.is_err());
        assert_eq!(current(), original);
        assert!(std::panic::catch_unwind(|| {
            let _ = stream.with_device(|| panic!("producer panicked"));
        })
        .is_err());
        assert_eq!(current(), original);
        // Also restore after a callback changes an initially matching device.
        let same = HandoffStream::new(original as u32).unwrap();
        same.with_device(|| {
            // SAFETY: target is an available CUDA device on this thread.
            unsafe { check(cudaSetDevice(target as c_int)).unwrap() };
        })
        .unwrap();
        assert_eq!(current(), original);
    }

    #[test]
    fn allocation_transfer_and_release_restore_the_calling_device() {
        let (mut original, mut count) = (0, 0);
        // SAFETY: both parameters are live integer outputs.
        unsafe {
            check(cudaGetDevice(&mut original)).unwrap();
            check(cudaGetDeviceCount(&mut count)).unwrap();
        }
        assert!(count > 0);
        let target = ((original + 1) % count) as u32;
        let tensor = Tensor::copy_from_cpu(
            &[1.0f32, 2.0],
            &[2],
            MemoryLocation::Cuda { device_id: target },
        )
        .unwrap();
        assert_eq!(tensor.copy_to_cpu::<f32>().unwrap(), [1.0, 2.0]);
        let mut current = -1;
        // SAFETY: current is a live output parameter.
        unsafe {
            check(cudaGetDevice(&mut current)).unwrap();
        }
        assert_eq!(current, original);
        drop(tensor);
        // SAFETY: current is a live output parameter.
        unsafe {
            check(cudaGetDevice(&mut current)).unwrap();
        }
        assert_eq!(current, original);
    }
}
