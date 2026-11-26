// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::engines::error::{Error, ErrorKind, Result};
use crate::MemoryLocation;
use std::{
    ffi::{c_int, c_void},
    ptr::{self, NonNull},
};

mod cuda_runtime_api;

const HOST_ALLOC_PORTABLE: u32 = 1;
const COPY_HOST_TO_DEVICE: c_int = 1;
const COPY_DEVICE_TO_HOST: c_int = 2;
const COPY_DEVICE_TO_DEVICE: c_int = 3;
const STREAM_NON_BLOCKING: u32 = 1;

/// A caller-owned CUDA stream handle (`cudaStream_t` or `CUstream`).
///
/// This wrapper never destroys the stream. Configure it through
/// [`crate::SessionOptions::cuda_streams`]; peers inherit the same handle.
#[derive(Debug, Clone, Copy)]
pub struct CudaStream {
    raw: *mut c_void,
}

// SAFETY: construction requires a live, externally retained CUDA handle. Native
// initialization rejects thread-local streams before any work is submitted.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Borrow a native stream without transferring ownership.
    /// Null and `cudaStreamLegacy` both select CUDA's legacy default stream.
    /// Native initialization rejects `cudaStreamPerThread` because sessions may
    /// execute on different host threads. Triton ignores configured streams.
    ///
    /// # Safety
    /// `stream` must be a valid CUDA stream handle or a CUDA default-stream
    /// sentinel. It must belong to the device's primary context, and its owner
    /// must retain it whenever any copy of this handle may be used, and until
    /// every session using it (including peers) is closed or dropped. Keep it
    /// alive through failed initialization as well. External
    /// submissions must respect the runtime's borrowed-buffer ownership rules.
    pub unsafe fn from_raw(stream: *mut c_void) -> Self {
        Self {
            // Use an explicit legacy handle even when a consumer was built with
            // per-thread default-stream semantics; null means "not supplied" in
            // our private backend ABI.
            raw: if stream.is_null() {
                ptr::without_provenance_mut(1)
            } else {
                stream
            },
        }
    }

    /// Return the borrowed CUDA handle, with null normalized to `cudaStreamLegacy`.
    pub fn as_raw(self) -> *mut c_void {
        self.raw
    }

    pub(crate) fn validate(self, device: u32) -> Result<()> {
        if self.raw as usize == 2 {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "load",
                "cudaStreamPerThread cannot be used by sessions that move between host threads",
            ));
        }
        validate(MemoryLocation::Cuda { device_id: device })?;
        let guard = Device::select(device)?;
        let api = guard.api;
        let mut event = ptr::null_mut();
        // SAFETY: construction guarantees a live stream. Recording an event
        // checks that it belongs to the selected runtime context/device, using
        // CUDA APIs available before cudaStreamGetDevice (CUDA 12.8).
        unsafe {
            api.check((api.cudaEventCreateWithFlags)(&mut event, 2))?; // cudaEventDisableTiming
            let recorded = api.check((api.cudaEventRecord)(event, self.raw));
            let destroyed = api.check((api.cudaEventDestroy)(event));
            recorded.map_err(|error| {
                Error::new(
                    ErrorKind::InvalidArgument,
                    "load",
                    format!("CUDA stream is incompatible with device {device}: {error}"),
                )
            })?;
            destroyed
        }
    }
}

pub(crate) fn validate(location: MemoryLocation) -> Result<()> {
    if let MemoryLocation::Cuda { device_id } = location {
        let api = cuda_runtime_api::load()?;
        let mut count = 0;
        // SAFETY: count is a live output parameter.
        unsafe {
            api.check((api.cudaGetDeviceCount)(&mut count))?;
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

pub(crate) struct Device {
    api: &'static cuda_runtime_api::CudaRuntimeApi,
    previous: c_int,
    changed: bool,
}

impl Device {
    pub(crate) fn select(device: u32) -> Result<Self> {
        Self::with_api(cuda_runtime_api::load()?, device)
    }

    fn with_api(api: &'static cuda_runtime_api::CudaRuntimeApi, device: u32) -> Result<Self> {
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
            api.check((api.cudaGetDevice)(&mut previous))?;
            if previous != device {
                api.check((api.cudaSetDevice)(device))?;
            }
        }
        Ok(Self {
            api,
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
                (self.api.cudaSetDevice)(self.previous);
            }
        }
    }
}

pub(crate) fn allocate(len: usize, location: MemoryLocation, zeroed: bool) -> Result<NonNull<u8>> {
    let api = cuda_runtime_api::load()?;
    let mut raw = ptr::null_mut();
    // SAFETY: raw is writable and the caller validated location and allocation size.
    unsafe {
        match location {
            MemoryLocation::PinnedCpu => {
                api.check((api.cudaHostAlloc)(&mut raw, len, HOST_ALLOC_PORTABLE))?;
                if zeroed {
                    ptr::write_bytes(raw.cast::<u8>(), 0, len);
                }
            }
            MemoryLocation::Cuda { device_id } => {
                let _device = Device::select(device_id)?;
                api.check((api.cudaMalloc)(&mut raw, len))?;
                if zeroed {
                    if let Err(error) = api
                        .check((api.cudaMemset)(raw, 0, len))
                        .and_then(|_| api.check((api.cudaDeviceSynchronize)()))
                    {
                        (api.cudaFree)(raw);
                        return Err(error);
                    }
                }
            }
            MemoryLocation::Cpu => unreachable!(),
        }
    }
    NonNull::new(raw.cast())
        .ok_or_else(|| Error::new(ErrorKind::Internal, "buffer", "CUDA returned null storage"))
}

pub(crate) unsafe fn free(ptr: NonNull<u8>, location: MemoryLocation) {
    let Some(api) = cuda_runtime_api::loaded() else {
        return;
    };
    // SAFETY: caller transfers the final ownership of this CUDA allocation.
    unsafe {
        match location {
            MemoryLocation::PinnedCpu => {
                (api.cudaFreeHost)(ptr.as_ptr().cast());
            }
            MemoryLocation::Cuda { device_id } => {
                if let Ok(_device) = Device::with_api(api, device_id) {
                    (api.cudaFree)(ptr.as_ptr().cast());
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
    let guard = Device::select(device)?;
    let api = guard.api;
    // SAFETY: caller supplies valid regions and the correct transfer direction.
    unsafe {
        api.check((api.cudaMemcpy)(
            dst,
            src,
            len,
            if to_device {
                COPY_HOST_TO_DEVICE
            } else {
                COPY_DEVICE_TO_HOST
            },
        ))?;
        api.check((api.cudaDeviceSynchronize)())
    }
}

pub(crate) fn synchronize(device: u32) -> Result<()> {
    let guard = Device::select(device)?;
    let api = guard.api;
    // SAFETY: synchronize work on the selected device before releasing borrowed storage.
    unsafe { api.check((api.cudaDeviceSynchronize)()) }
}

/// A nonblocking CUDA stream used to receive producer completion dependencies.
pub(crate) struct HandoffStream {
    stream: *mut c_void,
    device: u32,
    owned: bool,
}

// SAFETY: the stream is owned or caller-retained, and every
// operation selects its device. Destruction requires exclusive ownership.
unsafe impl Send for HandoffStream {}

unsafe impl Sync for HandoffStream {}

impl HandoffStream {
    /// Create a stream on the selected device without changing the caller's device.
    pub fn new(device: u32) -> std::result::Result<Self, crate::RuntimeError> {
        let guard = Device::select(device)?;
        let api = guard.api;
        let mut stream = ptr::null_mut();
        // SAFETY: stream is a live output parameter on the selected device.
        unsafe {
            api.check((api.cudaStreamCreateWithFlags)(
                &mut stream,
                STREAM_NON_BLOCKING,
            ))?;
        }
        Ok(Self {
            stream,
            device,
            owned: true,
        })
    }

    /// Borrow a stream already validated during native engine initialization.
    pub(crate) fn borrowed(device: u32, stream: CudaStream) -> Self {
        Self {
            stream: stream.as_raw(),
            device,
            owned: false,
        }
    }

    /// Invoke a producer on this stream's device, then restore the caller's device.
    /// The callback runs synchronously on the current host thread.
    pub fn with_device<T>(
        &self,
        producer: impl FnOnce() -> T,
    ) -> std::result::Result<T, crate::RuntimeError> {
        let mut device = Device::select(self.device)?;
        // A producer may change the current device even if select did not.
        device.changed = true;
        Ok(producer())
    }

    /// Copy ready storage and wait only for this stream, including on errors.
    /// # Safety
    /// Source/destination must be disjoint, live regions of `len` bytes on the
    /// declared devices. This stream must belong to the CUDA destination, or
    /// the source device for device-to-host copies.
    pub(crate) unsafe fn copy_ready(
        &self,
        dst: *mut c_void,
        target: MemoryLocation,
        src: *const c_void,
        source: MemoryLocation,
        len: usize,
    ) -> Result<()> {
        let guard = Device::select(self.device)?;
        let api = guard.api;
        // SAFETY: caller supplies valid regions and retains them through this wait.
        unsafe {
            let result = match (source, target) {
                (
                    MemoryLocation::Cuda { device_id: from },
                    MemoryLocation::Cuda { device_id: to },
                ) if from != to => api.check((api.cudaMemcpyPeerAsync)(
                    dst,
                    to as c_int,
                    src,
                    from as c_int,
                    len,
                    self.stream,
                )),
                (_, MemoryLocation::Cuda { .. }) => api.check((api.cudaMemcpyAsync)(
                    dst,
                    src,
                    len,
                    if matches!(source, MemoryLocation::Cuda { .. }) {
                        COPY_DEVICE_TO_DEVICE
                    } else {
                        COPY_HOST_TO_DEVICE
                    },
                    self.stream,
                )),
                (MemoryLocation::Cuda { .. }, _) => api.check((api.cudaMemcpyAsync)(
                    dst,
                    src,
                    len,
                    COPY_DEVICE_TO_HOST,
                    self.stream,
                )),
                _ => unreachable!("CPU copies do not require a CUDA stream"),
            };
            let completed = api.check((api.cudaStreamSynchronize)(self.stream));
            result.and(completed)
        }
    }

    /// Handle passed to a DLPack producer's stream argument.
    pub fn address(&self) -> usize {
        self.stream as usize
    }

    /// Wait for producer work before handing data to engine execution streams.
    pub fn synchronize(&self) -> std::result::Result<(), crate::RuntimeError> {
        let guard = Device::select(self.device)?;
        let api = guard.api;
        // SAFETY: this object retains or borrows a live stream on the selected device.
        unsafe { api.check((api.cudaStreamSynchronize)(self.stream)) }.map_err(Into::into)
    }
}

impl Drop for HandoffStream {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        let Some(api) = cuda_runtime_api::loaded() else {
            return;
        };
        if let Ok(_device) = Device::with_api(api, self.device) {
            // SAFETY: this object is the final owner of the stream.
            unsafe {
                (api.cudaStreamDestroy)(self.stream);
            }
        }
    }
}

pub(crate) unsafe fn copy_device(
    dst: *mut c_void,
    src: *const c_void,
    len: usize,
    source_device: u32,
    target_device: u32,
) -> Result<()> {
    let guard = Device::select(target_device)?;
    let api = guard.api;
    // SAFETY: caller supplies disjoint live allocations on the declared devices.
    unsafe {
        if source_device == target_device {
            api.check((api.cudaMemcpy)(dst, src, len, COPY_DEVICE_TO_DEVICE))?;
        } else {
            api.check((api.cudaMemcpyPeer)(
                dst,
                target_device as c_int,
                src,
                source_device as c_int,
                len,
            ))?;
        }
        api.check((api.cudaDeviceSynchronize)())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit_tests/memory/cuda.rs"]
mod tests;
