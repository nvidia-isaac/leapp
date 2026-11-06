// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Triton tensor types and request inputs.
use crate::dlpack::{BorrowedTensor, DLDataType};
use crate::engines::triton::native::{error::check, sys};
use crate::engines::{
    error::{Error, ErrorKind, Result},
    memory::dlpack::{layout, location},
};
use std::ffi::c_char;

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

#[cfg(test)]
#[path = "../../../../tests/unit_tests/triton/dlpack.rs"]
mod tests;
