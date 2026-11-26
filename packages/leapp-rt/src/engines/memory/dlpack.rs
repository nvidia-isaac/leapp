// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validated DLPack layouts, copies, and shared output ownership.

use super::Buffer;
use crate::dlpack::{sys::*, BorrowedTensor, ManagedTensor, SharedTensor};
use crate::engines::error::{Error, ErrorKind, Result};
use crate::{DataType, MemoryLocation};
use std::{ptr, sync::Arc};

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidArgument, "dlpack", message)
}

pub(crate) fn data_type(dtype: DLDataType) -> Result<DataType> {
    if dtype.lanes != 1 {
        return Err(invalid("DLPack vector lanes are unsupported"));
    }
    Ok(match (dtype.code, dtype.bits) {
        (0, 8) => DataType::Int8,
        (0, 16) => DataType::Int16,
        (0, 32) => DataType::Int32,
        (0, 64) => DataType::Int64,
        (1, 8) => DataType::Uint8,
        (1, 16) => DataType::Uint16,
        (1, 32) => DataType::Uint32,
        (1, 64) => DataType::Uint64,
        (2, 16) => DataType::Float16,
        (2, 32) => DataType::Float32,
        (2, 64) => DataType::Float64,
        (4, 16) => DataType::Bfloat16,
        (6, 8) => DataType::Bool,
        _ => return Err(invalid("unsupported DLPack dtype")),
    })
}

pub(crate) fn dl_type(dtype: DataType) -> DLDataType {
    let (code, bits) = match dtype {
        DataType::Int8 => (0, 8),
        DataType::Int16 => (0, 16),
        DataType::Int32 => (0, 32),
        DataType::Int64 => (0, 64),
        DataType::Uint8 => (1, 8),
        DataType::Uint16 => (1, 16),
        DataType::Uint32 => (1, 32),
        DataType::Uint64 => (1, 64),
        DataType::Float16 => (2, 16),
        DataType::Float32 => (2, 32),
        DataType::Float64 => (2, 64),
        DataType::Bfloat16 => (4, 16),
        DataType::Bool => (6, 8),
    };
    DLDataType {
        code,
        bits,
        lanes: 1,
    }
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

pub(crate) fn device(location: MemoryLocation) -> Result<DLDevice> {
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
    data_type(dtype)?;
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
pub(crate) fn layout(input: BorrowedTensor<'_>) -> Result<(&[i64], usize)> {
    layout_with_device(input, None)
}

/// The workspace already validated its execution device at initialization.
pub(crate) fn layout_on(input: BorrowedTensor<'_>, known: DLDevice) -> Result<(&[i64], usize)> {
    layout_with_device(input, Some(known))
}

/// Read a layout that has already passed the engine boundary.
/// # Safety
/// Metadata must retain its validated contiguous layout, dtype and byte size,
/// and shape pointers must outlive the returned slice.
pub(crate) unsafe fn layout_unchecked<'a>(tensor: DLTensor) -> (&'a [i64], usize) {
    let shape = if tensor.ndim == 0 {
        &[]
    } else {
        // SAFETY: the caller supplies validated, unchanged metadata.
        unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) }
    };
    let count = shape
        .iter()
        .fold(1usize, |count, &dim| count * dim as usize);
    let len = count * usize::from(tensor.dtype.bits / 8);
    (shape, len)
}

fn layout_with_device(
    input: BorrowedTensor<'_>,
    known: Option<DLDevice>,
) -> Result<(&[i64], usize)> {
    // SAFETY: the input borrow retains the descriptor's metadata for the returned slice.
    let result = unsafe { layout_metadata(*input.descriptor()) }?;
    let actual = input.descriptor().device;
    if !known.is_some_and(|device| {
        device.device_type == actual.device_type && device.device_id == actual.device_id
    }) {
        location(actual)?.validate()?;
    }
    Ok(result)
}

/// Check contiguous tensor metadata without probing device availability.
/// # Safety
/// Shape and stride pointers must cover the declared rank and outlive the returned slice.
pub(crate) unsafe fn layout_metadata<'a>(t: DLTensor) -> Result<(&'a [i64], usize)> {
    let n = usize::try_from(t.ndim).map_err(|_| invalid("negative dimension count"))?;
    if n > isize::MAX as usize / 8 || (n != 0 && t.shape.is_null()) {
        return Err(invalid("invalid shape descriptor"));
    }
    // SAFETY: the caller covers metadata lengths; zero-dimensional shapes need no pointer.
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
            // SAFETY: the caller guarantees ndim stride entries when non-null.
            if shape[i] > 1 && unsafe { *t.strides.add(i) } != expected {
                return Err(invalid("DLPack tensors must be C-contiguous"));
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
    Ok((shape, len))
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

/// # Safety
/// Dtype and shape must have been checked against this ready allocation's byte
/// size. All dimensions and the rank must be valid and unchanged since validation.
pub(crate) unsafe fn from_buffer(
    dtype: DLDataType,
    mut shape: Vec<i64>,
    buffer: Arc<Buffer>,
) -> Result<SharedTensor> {
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
pub fn copy(input: BorrowedTensor<'_>, target: Option<MemoryLocation>) -> Result<SharedTensor> {
    let t = input.descriptor();
    let (shape, len) = layout(input)?;
    let source = location(t.device)?;
    let target = target.unwrap_or(source);
    let buffer = Arc::new(Buffer::allocate_for_copy(len, target)?);
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
                    crate::engines::memory::cuda::copy_device(
                        buffer.ptr.as_ptr().cast(),
                        ptr,
                        len,
                        device_id,
                        dst,
                    )?
                }
                (MemoryLocation::Cuda { device_id }, _) => crate::engines::memory::cuda::copy(
                    buffer.ptr.as_ptr().cast(),
                    ptr,
                    len,
                    device_id,
                    false,
                )?,
                _ => buffer.copy_from(ptr)?,
            }
        }
    }
    // SAFETY: layout established the byte size; the completed copy initialized the allocation.
    unsafe { from_buffer(t.dtype, shape.to_vec(), buffer) }
}

#[cfg(test)]
#[path = "../../../tests/unit_tests/memory/dlpack.rs"]
mod tests;

/// Validated contiguous ranges overlap only within the same address space.
pub(crate) fn ranges_overlap(
    left: crate::dlpack::DLDevice,
    a: usize,
    a_len: usize,
    right: crate::dlpack::DLDevice,
    b: usize,
    b_len: usize,
) -> bool {
    let space = |device: crate::dlpack::DLDevice| {
        if device.device_type == crate::dlpack::kDLCUDA as i32 {
            (2, device.device_id)
        } else {
            (1, 0)
        }
    };
    a_len != 0 && b_len != 0 && space(left) == space(right) && a < b + b_len && b < a + a_len
}
