// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct DLPack/Triton conversion and private output ownership.
use super::Buffer;
use crate::dlpack::{sys::*, BorrowedTensor, ManagedTensor, SharedTensor};
use crate::engines::triton::{
    native::{error::check, sys},
    Error, ErrorKind, MemoryLocation, Result,
};
use std::{ffi::c_char, ptr, sync::Arc};

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidArgument, "dlpack", message)
}

pub(crate) fn dtype_to_native(d: DLDataType) -> Result<u32> {
    if d.lanes != 1 {
        return Err(invalid("DLPack vector lanes are unsupported"));
    }
    Ok(match (d.code, d.bits) {
        (0, 8) => sys::TRITONSERVER_TYPE_INT8,
        (0, 16) => sys::TRITONSERVER_TYPE_INT16,
        (0, 32) => sys::TRITONSERVER_TYPE_INT32,
        (0, 64) => sys::TRITONSERVER_TYPE_INT64,
        (1, 8) => sys::TRITONSERVER_TYPE_UINT8,
        (1, 16) => sys::TRITONSERVER_TYPE_UINT16,
        (1, 32) => sys::TRITONSERVER_TYPE_UINT32,
        (1, 64) => sys::TRITONSERVER_TYPE_UINT64,
        (2, 16) => sys::TRITONSERVER_TYPE_FP16,
        (2, 32) => sys::TRITONSERVER_TYPE_FP32,
        (2, 64) => sys::TRITONSERVER_TYPE_FP64,
        (4, 16) => sys::TRITONSERVER_TYPE_BF16,
        (6, 8) => sys::TRITONSERVER_TYPE_BOOL,
        _ => return Err(invalid("unsupported DLPack dtype")),
    })
}
pub(crate) fn dtype_from_native(d: u32) -> Result<DLDataType> {
    let (code, bits) = match d {
        sys::TRITONSERVER_TYPE_INT8 => (0, 8),
        sys::TRITONSERVER_TYPE_INT16 => (0, 16),
        sys::TRITONSERVER_TYPE_INT32 => (0, 32),
        sys::TRITONSERVER_TYPE_INT64 => (0, 64),
        sys::TRITONSERVER_TYPE_UINT8 => (1, 8),
        sys::TRITONSERVER_TYPE_UINT16 => (1, 16),
        sys::TRITONSERVER_TYPE_UINT32 => (1, 32),
        sys::TRITONSERVER_TYPE_UINT64 => (1, 64),
        sys::TRITONSERVER_TYPE_FP16 => (2, 16),
        sys::TRITONSERVER_TYPE_FP32 => (2, 32),
        sys::TRITONSERVER_TYPE_FP64 => (2, 64),
        sys::TRITONSERVER_TYPE_BF16 => (4, 16),
        sys::TRITONSERVER_TYPE_BOOL => (6, 8),
        _ => return Err(invalid("unsupported Triton output dtype")),
    };
    Ok(DLDataType {
        code,
        bits,
        lanes: 1,
    })
}
pub fn location(device: DLDevice) -> Result<MemoryLocation> {
    match (device.device_type, device.device_id) {
        (1, 0) => Ok(MemoryLocation::Cpu),
        (3, 0) => Ok(MemoryLocation::PinnedCpu),
        (2, id) if id >= 0 => Ok(MemoryLocation::Cuda {
            device_id: id as u32,
        }),
        _ => Err(invalid("unsupported DLPack device")),
    }
}
fn device(location: MemoryLocation) -> Result<DLDevice> {
    let (device_type, device_id) = match location {
        MemoryLocation::Cpu => (1, 0),
        MemoryLocation::PinnedCpu => (3, 0),
        MemoryLocation::Cuda { device_id } => (
            2,
            i32::try_from(device_id).map_err(|_| invalid("device ID overflow"))?,
        ),
    };
    Ok(DLDevice {
        device_type,
        device_id,
    })
}
pub(crate) fn byte_size(dtype: DLDataType, shape: &[i64]) -> Result<usize> {
    dtype_to_native(dtype)?;
    let mut count = 1usize;
    for &d in shape {
        count = count
            .checked_mul(usize::try_from(d).map_err(|_| invalid("negative dimension"))?)
            .ok_or_else(|| invalid("tensor size overflow"))?;
    }
    count
        .checked_mul(usize::from(dtype.bits / 8))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid("tensor byte size overflow"))
}

/// Validate metadata once at the engine boundary. The borrow supplies pointer validity.
fn layout(input: BorrowedTensor<'_>) -> Result<(&[i64], usize)> {
    let t = input.descriptor();
    let n = usize::try_from(t.ndim).map_err(|_| invalid("negative dimension count"))?;
    if n > isize::MAX as usize / 8 || (n != 0 && t.shape.is_null()) {
        return Err(invalid("invalid shape descriptor"));
    }
    // SAFETY: BorrowedTensor's contract covers metadata lengths; zero-dimensional shapes need no pointer.
    let shape = if n == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(t.shape, n) }
    };
    let len = byte_size(t.dtype, shape)?;
    if len != 0 && t.data.is_null() {
        return Err(invalid("null tensor storage"));
    }
    if !t.strides.is_null() && len != 0 {
        let mut expected = 1i64;
        for i in (0..n).rev() {
            // SAFETY: BorrowedTensor guarantees ndim stride entries when non-null.
            if shape[i] > 1 && unsafe { *t.strides.add(i) } != expected {
                return Err(invalid("DLPack inputs must be C-contiguous"));
            }
            expected = expected
                .checked_mul(shape[i])
                .ok_or_else(|| invalid("stride overflow"))?;
        }
    }
    let offset = usize::try_from(t.byte_offset).map_err(|_| invalid("byte offset overflow"))?;
    (t.data as usize)
        .checked_add(offset)
        .and_then(|v| v.checked_add(len))
        .ok_or_else(|| invalid("tensor address overflow"))?;
    location(t.device)?.validate()?;
    Ok((shape, len))
}

pub(crate) unsafe fn append_input(
    request: *mut sys::TRITONSERVER_InferenceRequest,
    name: *const c_char,
    input: BorrowedTensor<'_>,
) -> Result<()> {
    let t = input.descriptor();
    let (shape, len) = layout(input)?;
    let (memory, device) = location(t.device)?.native();
    // SAFETY: caller owns request/name; the input borrow retains ready storage until request release.
    unsafe {
        check(
            sys::TRITONSERVER_InferenceRequestAddInput(
                request,
                name,
                dtype_to_native(t.dtype)?,
                shape.as_ptr(),
                shape.len() as u64,
            ),
            "execute",
        )?;
        check(
            sys::TRITONSERVER_InferenceRequestAppendInputData(
                request,
                name,
                t.data
                    .cast::<u8>()
                    .wrapping_add(t.byte_offset as usize)
                    .cast(),
                len,
                memory,
                device,
            ),
            "execute",
        )
    }
}

struct Output {
    _shape: Vec<i64>,
    _buffer: Arc<Buffer>,
}
unsafe extern "C" fn delete(p: *mut DLManagedTensorVersioned) {
    // SAFETY: from_buffer allocated these two boxes and this is their sole deleter.
    unsafe {
        let p = Box::from_raw(p);
        drop(Box::from_raw(p.manager_ctx.cast::<Output>()));
    }
}
pub(crate) fn from_buffer(
    dtype: DLDataType,
    mut shape: Vec<i64>,
    buffer: Arc<Buffer>,
) -> Result<SharedTensor> {
    if byte_size(dtype, &shape)? != buffer.len {
        return Err(invalid("output byte size mismatch"));
    }
    let descriptor = DLTensor {
        data: if buffer.len == 0 {
            ptr::null_mut()
        } else {
            buffer.ptr.as_ptr().cast()
        },
        device: device(buffer.location)?,
        ndim: i32::try_from(shape.len()).map_err(|_| invalid("too many dimensions"))?,
        dtype,
        shape: shape.as_mut_ptr(),
        strides: ptr::null_mut(),
        byte_offset: 0,
    };
    let exposed_buffer = buffer.clone();
    let copy_buffer = buffer.clone();
    let context = Box::into_raw(Box::new(Output {
        _shape: shape,
        _buffer: buffer,
    }))
    .cast();
    let p = Box::into_raw(Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: context,
        deleter: Some(delete),
        flags: 0,
        dl_tensor: descriptor,
    }));
    // SAFETY: callers provide initialized, completed output storage; native cleanup
    // is device-aware and thread independent. The copy callback retains the buffer,
    // checks the destination size, and completes the transfer before returning.
    unsafe {
        SharedTensor::new(
            ManagedTensor::from_versioned(p).unwrap(),
            Some(Box::new(move || exposed_buffer.expose())),
            Some(Box::new(move |destination| {
                if destination.len() != copy_buffer.len {
                    return Err("CPU copy size does not match allocation".into());
                }
                // SAFETY: destination is exclusive CPU storage of the checked size;
                // the owner excludes writable exposure until this transfer finishes.
                copy_buffer
                    .copy_to(destination.as_mut_ptr().cast())
                    .map_err(|error| error.to_string())
            })),
        )
        .map_err(invalid)
    }
}

/// Engine allocation used for initial state and isolation of writable public outputs.
/// This is deliberately not a tensor method or framework conversion API.
#[cfg(test)]
pub fn copy(input: BorrowedTensor<'_>, target: Option<MemoryLocation>) -> Result<SharedTensor> {
    copy_with(input, target, |len, location| {
        Ok(Arc::new(Buffer::allocate_for_copy(len, location)?))
    })
}

pub(crate) fn copy_with(
    input: BorrowedTensor<'_>,
    target: Option<MemoryLocation>,
    allocate: impl FnOnce(usize, MemoryLocation) -> Result<Arc<Buffer>>,
) -> Result<SharedTensor> {
    let t = input.descriptor();
    let (shape, len) = layout(input)?;
    let source = location(t.device)?;
    let target = target.unwrap_or(source);
    let buffer = allocate(len, target)?;
    let ptr = t
        .data
        .cast::<u8>()
        .wrapping_add(t.byte_offset as usize)
        .cast();
    if len != 0 {
        // SAFETY: validated borrowed source and new, disjoint output allocation.
        unsafe {
            match (source, target) {
                (MemoryLocation::Cuda { device_id }, MemoryLocation::Cuda { device_id: dst }) => {
                    crate::engines::triton::memory::cuda::copy_device(
                        buffer.ptr.as_ptr().cast(),
                        ptr,
                        len,
                        device_id,
                        dst,
                    )?
                }
                (MemoryLocation::Cuda { device_id }, _) => {
                    crate::engines::triton::memory::cuda::copy(
                        buffer.ptr.as_ptr().cast(),
                        ptr,
                        len,
                        device_id,
                        false,
                    )?
                }
                _ => buffer.copy_from(ptr)?,
            }
        }
    }
    from_buffer(t.dtype, shape.to_vec(), buffer)
}

#[cfg(test)]
#[path = "../../../../tests/unit_tests/triton/dlpack.rs"]
mod tests;
