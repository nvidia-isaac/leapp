// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::mem::{align_of, size_of};

mod private {
    pub trait Sealed {}
}

/// Rust numeric primitives with a DLPack dtype and no invalid bit patterns.
/// Sealed to prevent incorrect dtype declarations. Half floats and booleans use
/// the raw descriptor API.
pub trait CpuElement: private::Sealed + Copy + Default + Send + Sync {
    const DTYPE: DLDataType;
}

macro_rules! elements {
    ($($ty:ty => $code:expr),* $(,)?) => {$ (
        impl private::Sealed for $ty {}
        impl CpuElement for $ty {
            const DTYPE: DLDataType = DLDataType {
                code: $code as u8, bits: (size_of::<Self>() * 8) as u8, lanes: 1,
            };
        }
    )*};
}
elements!(
    u8 => kDLUInt, u16 => kDLUInt, u32 => kDLUInt, u64 => kDLUInt,
    i8 => kDLInt, i16 => kDLInt, i32 => kDLInt, i64 => kDLInt,
    f32 => kDLFloat, f64 => kDLFloat,
);

fn element_count(shape: &[i64]) -> Result<usize, &'static str> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(usize::try_from(dimension).map_err(|_| "negative dimension")?)
            .ok_or("tensor size overflow")
    })
}

impl<'a> BorrowedTensor<'a> {
    /// Borrow contiguous CPU storage without copying. The values and shape remain
    /// immutably borrowed for the tensor's lifetime. An empty shape is a scalar.
    pub fn cpu<T: CpuElement>(values: &'a [T], shape: &'a [i64]) -> Result<Self, &'static str> {
        let ndim = i32::try_from(shape.len()).map_err(|_| "too many dimensions")?;
        if element_count(shape)? != values.len() {
            return Err("shape does not match slice length");
        }
        Ok(Self {
            descriptor: DLTensor {
                data: values.as_ptr().cast_mut().cast(),
                device: DLDevice {
                    device_type: kDLCPU as i32,
                    device_id: 0,
                },
                ndim,
                dtype: T::DTYPE,
                shape: shape.as_ptr().cast_mut(),
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
            lifetime: PhantomData,
        })
    }

    /// Read a contiguous CPU tensor with the requested primitive type.
    /// The borrow already guarantees ready, immutable storage and valid pointers.
    pub fn as_cpu_slice<T: CpuElement>(self) -> Result<&'a [T], &'static str> {
        let t = self.descriptor();
        if t.device.device_type != kDLCPU as i32 && t.device.device_type != kDLCUDAHost as i32 {
            return Err("tensor is not in CPU memory");
        }
        let (data, count) = self.typed_layout::<T>()?;
        if count == 0 {
            return Ok(&[]);
        }
        // SAFETY: this borrow guarantees initialized, immutable CPU storage for 'a;
        // typed_layout checks dtype, contiguity, alignment, and length.
        Ok(unsafe { std::slice::from_raw_parts(data, count) })
    }

    fn typed_layout<T: CpuElement>(self) -> Result<(*const T, usize), &'static str> {
        let t = self.descriptor();
        if (t.dtype.code, t.dtype.bits, t.dtype.lanes) != (T::DTYPE.code, T::DTYPE.bits, 1) {
            return Err("tensor dtype does not match slice element type");
        }
        let ndim = usize::try_from(t.ndim).map_err(|_| "negative dimension count")?;
        if ndim > isize::MAX as usize / size_of::<i64>() || (ndim != 0 && t.shape.is_null()) {
            return Err("invalid shape descriptor");
        }
        // SAFETY: the tensor borrow guarantees valid, immutable metadata.
        let shape = if ndim == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(t.shape, ndim) }
        };
        let count = element_count(shape)?;
        let bytes = count
            .checked_mul(size_of::<T>())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or("tensor byte size overflow")?;
        if count == 0 {
            return Ok((NonNull::<T>::dangling().as_ptr(), 0));
        }
        if !t.strides.is_null() {
            let mut expected = 1i64;
            for i in (0..ndim).rev() {
                // SAFETY: the tensor borrow guarantees ndim valid stride entries.
                if shape[i] > 1 && unsafe { *t.strides.add(i) } != expected {
                    return Err("tensor is not C-contiguous");
                }
                expected = expected.checked_mul(shape[i]).ok_or("stride overflow")?;
            }
        }
        let offset = usize::try_from(t.byte_offset).map_err(|_| "byte offset overflow")?;
        let address = (t.data as usize)
            .checked_add(offset)
            .ok_or("tensor address overflow")?;
        address
            .checked_add(bytes)
            .ok_or("tensor address overflow")?;
        if t.data.is_null() || address % align_of::<T>() != 0 {
            return Err("null or unaligned tensor storage");
        }
        Ok((t.data.cast::<u8>().wrapping_add(offset).cast(), count))
    }
}

impl SharedTensor {
    /// Copy a contiguous output into an owned Rust vector, including device
    /// storage when the engine supplies a synchronous transfer callback.
    ///
    /// Rejects storage previously exposed through `descriptor()` or `export()`
    /// on any clone: a foreign consumer could still be writing it. Copies and
    /// first exposure are synchronized; copying does not prevent pool reuse.
    pub fn to_cpu_vec<T: CpuElement>(&self) -> Result<Vec<T>, String> {
        let exposed = self.0.exposed.lock().unwrap_or_else(|e| e.into_inner());
        if *exposed {
            return Err("tensor storage has been exposed to a writable consumer".into());
        }
        // SAFETY: construction guarantees ready storage; the lock excludes
        // first exposure until the copy completes, and no pointers escape.
        let borrowed = unsafe { BorrowedTensor::new(self.0.owner.descriptor()?) };
        let values = if let Some(copy) = &self.0.copy_to_cpu {
            let (_, count) = borrowed.typed_layout::<T>()?;
            let mut values = vec![T::default(); count];
            // SAFETY: CpuElement is sealed to padding-free primitives accepting
            // every bit pattern. The callback cannot retain this mutable borrow.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), count * size_of::<T>())
            };
            copy(bytes)?;
            values
        } else {
            borrowed.as_cpu_slice::<T>()?.to_vec()
        };
        drop(exposed);
        Ok(values)
    }
}
