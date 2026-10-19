// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{sys, Error, ErrorKind, Result};
use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    ffi::c_void,
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// Fixed-width numeric types understood by Triton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DataType {
    Bool = sys::TRITONSERVER_TYPE_BOOL,
    Uint8 = sys::TRITONSERVER_TYPE_UINT8,
    Uint16 = sys::TRITONSERVER_TYPE_UINT16,
    Uint32 = sys::TRITONSERVER_TYPE_UINT32,
    Uint64 = sys::TRITONSERVER_TYPE_UINT64,
    Int8 = sys::TRITONSERVER_TYPE_INT8,
    Int16 = sys::TRITONSERVER_TYPE_INT16,
    Int32 = sys::TRITONSERVER_TYPE_INT32,
    Int64 = sys::TRITONSERVER_TYPE_INT64,
    Float16 = sys::TRITONSERVER_TYPE_FP16,
    Float32 = sys::TRITONSERVER_TYPE_FP32,
    Float64 = sys::TRITONSERVER_TYPE_FP64,
    Bfloat16 = sys::TRITONSERVER_TYPE_BF16,
}

impl DataType {
    pub(crate) fn from_native(value: u32) -> Result<Self> {
        match value {
            sys::TRITONSERVER_TYPE_BOOL => Ok(Self::Bool),
            sys::TRITONSERVER_TYPE_UINT8 => Ok(Self::Uint8),
            sys::TRITONSERVER_TYPE_UINT16 => Ok(Self::Uint16),
            sys::TRITONSERVER_TYPE_UINT32 => Ok(Self::Uint32),
            sys::TRITONSERVER_TYPE_UINT64 => Ok(Self::Uint64),
            sys::TRITONSERVER_TYPE_INT8 => Ok(Self::Int8),
            sys::TRITONSERVER_TYPE_INT16 => Ok(Self::Int16),
            sys::TRITONSERVER_TYPE_INT32 => Ok(Self::Int32),
            sys::TRITONSERVER_TYPE_INT64 => Ok(Self::Int64),
            sys::TRITONSERVER_TYPE_FP16 => Ok(Self::Float16),
            sys::TRITONSERVER_TYPE_FP32 => Ok(Self::Float32),
            sys::TRITONSERVER_TYPE_FP64 => Ok(Self::Float64),
            sys::TRITONSERVER_TYPE_BF16 => Ok(Self::Bfloat16),
            _ => Err(Error::new(
                ErrorKind::Unsupported,
                "execute",
                "non-numeric Triton output",
            )),
        }
    }

    /// Number of bytes used by one scalar.
    pub fn byte_width(self) -> usize {
        match self {
            Self::Bool | Self::Uint8 | Self::Int8 => 1,
            Self::Uint16 | Self::Int16 | Self::Float16 | Self::Bfloat16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Primitive numeric values whose every bit pattern is valid. This trait is sealed.
pub trait TensorElement: sealed::Sealed + Copy + Default + Send + Sync + 'static {
    /// Tensor scalar type corresponding to this primitive.
    const DTYPE: DataType;
}

macro_rules! elements {
    ($($ty:ty => $dtype:ident),* $(,)?) => { $(
        impl sealed::Sealed for $ty {}

        impl TensorElement for $ty { const DTYPE: DataType = DataType::$dtype; }
    )* };
}

elements!(u8=>Uint8, u16=>Uint16, u32=>Uint32, u64=>Uint64,
          i8=>Int8, i16=>Int16, i32=>Int32, i64=>Int64, f32=>Float32, f64=>Float64);

/// Storage location, independent of a model's execution device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MemoryLocation {
    #[default]
    Cpu,
    PinnedCpu,
    Cuda {
        device_id: u32,
    },
}

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

        crate::cuda::validate(self)
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidArgument, "buffer", message)
}

pub(crate) fn byte_size(dtype: DataType, shape: &[i64]) -> Result<usize> {
    let mut elements = 1usize;
    for &dim in shape {
        let dim =
            usize::try_from(dim).map_err(|_| invalid("tensor dimensions must be nonnegative"))?;
        elements = elements
            .checked_mul(dim)
            .ok_or_else(|| invalid("tensor size overflow"))?;
    }
    let size = elements
        .checked_mul(dtype.byte_width())
        .ok_or_else(|| invalid("tensor byte size overflow"))?;
    if size > isize::MAX as usize {
        return Err(invalid("tensor is too large for a Rust slice"));
    }
    Ok(size)
}

/// Contiguous input borrowed for the entire synchronous inference call.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    pub(crate) dtype: DataType,
    pub(crate) shape: &'a [i64],
    pub(crate) ptr: *const c_void,
    pub(crate) len: usize,
    pub(crate) location: MemoryLocation,
    borrow: PhantomData<&'a [u8]>,
}

impl<'a> TensorView<'a> {
    /// Borrow an externally owned contiguous tensor.
    ///
    /// # Safety
    /// The pointer must cover the computed byte count in the declared location,
    /// be ready for reads on all streams, and remain alive and unmodified for
    /// the entire returned lifetime. Shape metadata must remain valid as well.
    pub unsafe fn from_raw_parts(
        ptr: *const c_void,
        dtype: DataType,
        shape: &'a [i64],
        location: MemoryLocation,
    ) -> Result<Self> {
        let len = byte_size(dtype, shape)?;
        location.validate()?;
        if len != 0 && ptr.is_null() {
            return Err(invalid("nonempty tensor has a null pointer"));
        }
        Ok(Self {
            dtype,
            shape,
            ptr,
            len,
            location,
            borrow: PhantomData,
        })
    }

    /// Borrow foreign storage with an explicit allocation byte count.
    ///
    /// # Safety
    /// The pointer must satisfy the same contract as `from_raw_parts`.
    pub unsafe fn from_raw_buffer(
        ptr: *const c_void,
        byte_count: usize,
        dtype: DataType,
        shape: &'a [i64],
        location: MemoryLocation,
    ) -> Result<Self> {
        if byte_size(dtype, shape)? != byte_count {
            return Err(invalid("shape does not match input byte count"));
        }
        // SAFETY: the caller guarantees storage and readiness for this borrow.
        unsafe { Self::from_raw_parts(ptr, dtype, shape, location) }
    }

    /// Borrow typed CPU storage with an exact shape/element-count match.
    pub fn cpu<T: TensorElement>(data: &'a [T], shape: &'a [i64]) -> Result<Self> {
        let len = std::mem::size_of_val(data);
        if byte_size(T::DTYPE, shape)? != len {
            return Err(invalid("shape does not match input byte count"));
        }
        Ok(Self {
            dtype: T::DTYPE,
            shape,
            ptr: data.as_ptr().cast(),
            len,
            location: MemoryLocation::Cpu,
            borrow: PhantomData,
        })
    }

    /// Borrow packed numeric bytes, including float16, bfloat16, and Boolean data.
    /// Boolean values must be encoded as 0 or 1.
    pub fn cpu_bytes(data: &'a [u8], dtype: DataType, shape: &'a [i64]) -> Result<Self> {
        if byte_size(dtype, shape)? != data.len() {
            return Err(invalid("shape does not match input byte count"));
        }
        if dtype == DataType::Bool && data.iter().any(|&v| v > 1) {
            return Err(invalid("Boolean data must be 0 or 1"));
        }
        Ok(Self {
            dtype,
            shape,
            ptr: data.as_ptr().cast(),
            len: data.len(),
            location: MemoryLocation::Cpu,
            borrow: PhantomData,
        })
    }
}

// SAFETY: safe constructors borrow immutable storage. The raw constructor
// requires the caller to guarantee the same lifetime and read-only contract,
// including readiness for cross-thread access.
unsafe impl Send for TensorView<'_> {}

unsafe impl Sync for TensorView<'_> {}

/// Immutable tensor storage retained independently of the Triton response.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) dtype: DataType,
    pub(crate) shape: Vec<i64>,
    pub(crate) buffer: Arc<Buffer>,
}

impl Tensor {
    /// Copy typed CPU values into the requested memory location; returns ready data.
    pub fn copy_from_cpu<T: TensorElement>(
        data: &[T],
        shape: &[i64],
        location: MemoryLocation,
    ) -> Result<Self> {
        Self::from_cpu_view(TensorView::cpu(data, shape)?, location)
    }

    /// Copy packed numeric bytes, including float16, bfloat16 and Boolean data.
    pub fn copy_from_cpu_bytes(
        data: &[u8],
        dtype: DataType,
        shape: &[i64],
        location: MemoryLocation,
    ) -> Result<Self> {
        Self::from_cpu_view(TensorView::cpu_bytes(data, dtype, shape)?, location)
    }

    fn from_cpu_view(view: TensorView<'_>, location: MemoryLocation) -> Result<Self> {
        let buffer = Buffer::allocate(view.len, location)?;
        // SAFETY: callers supply a CPU view covering view.len bytes; this
        // allocation is not shared until the synchronous copy has finished.
        unsafe {
            buffer.copy_from(view.ptr)?;
        }
        Ok(Self {
            dtype: view.dtype,
            shape: view.shape.to_vec(),
            buffer: Arc::new(buffer),
        })
    }

    /// Borrow this tensor as an input without copying its storage.
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            dtype: self.dtype,
            shape: &self.shape,
            ptr: self.buffer.ptr.as_ptr().cast(),
            len: self.buffer.len,
            location: self.buffer.location,
            borrow: PhantomData,
        }
    }

    /// Number of bytes in the retained allocation.
    pub fn byte_size(&self) -> usize {
        self.buffer.len
    }

    /// Read-only storage address, valid while this allocation is retained.
    /// Permanently excludes this allocation from output-pool reuse. This does not
    /// extend pointer lifetime or synchronize later work by foreign consumers.
    pub fn data_ptr(&self) -> *const c_void {
        // Foreign GPU consumers can enqueue reads before dropping their owner.
        // Exclude such allocations from recycling rather than assuming completion.
        self.buffer.exposed.store(true, Ordering::Release);
        self.buffer.ptr.as_ptr().cast()
    }

    /// Allocate an independent copy in the same location, ready on return.
    pub fn deep_copy(&self) -> Result<Self> {
        let buffer = Buffer::allocate(self.buffer.len, self.location())?;
        if buffer.len != 0 {
            // SAFETY: both regions are owned, disjoint, and have identical sizes.
            unsafe {
                if let MemoryLocation::Cuda { device_id } = self.location() {
                    crate::cuda::copy_device(
                        buffer.ptr.as_ptr().cast(),
                        self.buffer.ptr.as_ptr().cast(),
                        buffer.len,
                        device_id,
                    )?;
                } else {
                    buffer.copy_from(self.buffer.ptr.as_ptr().cast())?;
                }
            }
        }
        Ok(Self {
            dtype: self.dtype,
            shape: self.shape.clone(),
            buffer: Arc::new(buffer),
        })
    }

    /// Scalar type of this tensor.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Concrete tensor dimensions.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Location of the owned allocation.
    pub fn location(&self) -> MemoryLocation {
        self.buffer.location
    }

    /// Borrow typed host data, rejecting device storage and a mismatched type.
    pub fn as_cpu_slice<T: TensorElement>(&self) -> Result<&[T]> {
        if matches!(self.location(), MemoryLocation::Cuda { .. }) {
            return Err(invalid("CUDA data requires an explicit host copy"));
        }
        if self.dtype != T::DTYPE || self.buffer.ptr.as_ptr().align_offset(align_of::<T>()) != 0 {
            return Err(invalid(
                "tensor dtype or alignment does not match the requested slice",
            ));
        }
        // SAFETY: allocations are aligned, initialized before publication, and owned for
        // this borrow. Sealed TensorElement implementations admit every bit pattern.
        Ok(unsafe {
            std::slice::from_raw_parts(
                self.buffer.ptr.as_ptr().cast(),
                self.buffer.len / size_of::<T>(),
            )
        })
    }

    /// Borrow packed host bytes, rejecting CUDA storage.
    pub fn as_cpu_bytes(&self) -> Result<&[u8]> {
        if matches!(self.location(), MemoryLocation::Cuda { .. }) {
            return Err(invalid("CUDA data requires an explicit host copy"));
        }
        // SAFETY: the buffer is initialized before publication and remains owned
        // for this borrow. Every byte pattern is valid, including packed half data.
        Ok(unsafe { std::slice::from_raw_parts(self.buffer.ptr.as_ptr(), self.buffer.len) })
    }

    /// Copy packed data to CPU, preserving its representation and synchronizing CUDA.
    pub fn copy_to_cpu_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.buffer.len];
        // SAFETY: bytes provides exactly the allocation's writable byte count.
        unsafe {
            self.buffer.copy_to(bytes.as_mut_ptr().cast())?;
        }
        Ok(bytes)
    }

    /// Copy data to a typed CPU vector, synchronizing device transfers.
    pub fn copy_to_cpu<T: TensorElement>(&self) -> Result<Vec<T>> {
        if self.dtype != T::DTYPE {
            return Err(invalid("tensor dtype does not match requested host copy"));
        }
        let mut result = vec![T::default(); self.buffer.len / size_of::<T>()];
        // SAFETY: the destination has exactly the allocation's byte count.
        unsafe {
            self.buffer.copy_to(result.as_mut_ptr().cast())?;
        }
        Ok(result)
    }
}

pub(crate) struct Buffer {
    pub(crate) ptr: NonNull<u8>,
    pub(crate) len: usize,
    pub(crate) location: MemoryLocation,
    exposed: AtomicBool,
}

// SAFETY: Buffer is private and filled before publication as Tensor. Only an
// unexposed allocation with no other strong or weak owners can be reserved for
// another write under the pool mutex. Published tensors expose immutable reads
// and copies; CUDA operations select the allocation's device.
unsafe impl Send for Buffer {}

unsafe impl Sync for Buffer {}

impl Buffer {
    pub(crate) fn can_reuse(&self) -> bool {
        !self.exposed.load(Ordering::Acquire)
    }

    pub(crate) fn allocate(len: usize, location: MemoryLocation) -> Result<Self> {
        location.validate()?;
        let layout = Layout::from_size_align(len, align_of::<u64>())
            .map_err(|_| invalid("invalid allocation size"))?;
        let ptr = if len == 0 {
            NonNull::<u64>::dangling().cast()
        } else if location == MemoryLocation::Cpu {
            // SAFETY: layout is nonzero and valid. Zero initialization permits safe
            // byte reads even if a faulty backend leaves part of an output unwritten.
            NonNull::new(unsafe { alloc_zeroed(layout) })
                .ok_or_else(|| Error::new(ErrorKind::Internal, "buffer", "CPU allocation failed"))?
        } else {
            crate::cuda::allocate(len, location)?
        };
        Ok(Self {
            ptr,
            len,
            location,
            exposed: AtomicBool::new(false),
        })
    }

    unsafe fn copy_from(&self, source: *const c_void) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }

        if let MemoryLocation::Cuda { device_id } = self.location {
            // SAFETY: caller provides a readable source; self owns the destination.
            return unsafe {
                crate::cuda::copy(self.ptr.as_ptr().cast(), source, self.len, device_id, true)
            };
        }
        // SAFETY: caller provides len readable bytes, disjoint from this allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(source.cast::<u8>(), self.ptr.as_ptr(), self.len);
        }
        Ok(())
    }

    unsafe fn copy_to(&self, destination: *mut c_void) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }

        if let MemoryLocation::Cuda { device_id } = self.location {
            // SAFETY: caller provides a writable destination; self owns the source.
            return unsafe {
                crate::cuda::copy(
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
                crate::cuda::free(self.ptr, self.location);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_views_reject_shape_and_size_mismatches() {
        assert!(TensorView::cpu(&[1.0f32], &[2]).is_err());
        assert!(TensorView::cpu(&[1.0f32], &[-1]).is_err());
        assert!(TensorView::cpu(&[1.0f32], &[i64::MAX, 4]).is_err());
        assert!(TensorView::cpu(&[1.0f32], &[]).is_ok());
        assert!(TensorView::cpu::<f32>(&[], &[0]).is_ok());
        assert!(TensorView::cpu_bytes(&[2], DataType::Bool, &[1]).is_err());
        assert!(TensorView::cpu_bytes(&[0, 1], DataType::Bool, &[2]).is_ok());
        assert!(TensorView::cpu_bytes(&[0, 0], DataType::Float16, &[1]).is_ok());
    }

    #[test]
    fn owned_host_tensors_are_aligned_typed_and_share_storage() {
        let tensor = Tensor::copy_from_cpu(&[1.0f64, 2.0], &[2], MemoryLocation::Cpu).unwrap();
        let copy = tensor.clone();
        assert!(Arc::ptr_eq(&tensor.buffer, &copy.buffer));
        drop(tensor);
        assert_eq!(copy.as_cpu_slice::<f64>().unwrap(), [1.0, 2.0]);
        assert!(copy.as_cpu_slice::<f32>().is_err());
        assert!(copy.copy_to_cpu::<u64>().is_err());
        let empty = Tensor::copy_from_cpu::<u64>(&[], &[0], MemoryLocation::Cpu).unwrap();
        assert!(empty.as_cpu_slice::<u64>().unwrap().is_empty());
    }

    #[test]
    fn packed_values_round_trip_without_reinterpreting_half_or_boolean_types() {
        let locations = vec![
            MemoryLocation::Cpu,
            MemoryLocation::PinnedCpu,
            MemoryLocation::Cuda { device_id: 0 },
        ];
        for location in locations {
            for (dtype, bytes, shape) in [
                (DataType::Bool, vec![0, 1], vec![2]),
                (DataType::Float16, vec![0, 60], vec![1]),
                (DataType::Bfloat16, vec![128, 63], vec![1]),
            ] {
                let tensor = Tensor::copy_from_cpu_bytes(&bytes, dtype, &shape, location).unwrap();
                assert_eq!(tensor.copy_to_cpu_bytes().unwrap(), bytes);
                assert_eq!(tensor.dtype(), dtype);
                if matches!(location, MemoryLocation::Cuda { .. }) {
                    assert!(tensor.as_cpu_bytes().is_err());
                } else {
                    assert_eq!(tensor.as_cpu_bytes().unwrap(), bytes);
                }
            }
        }
    }
}
