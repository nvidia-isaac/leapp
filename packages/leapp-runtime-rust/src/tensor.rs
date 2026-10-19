// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Safe public tensor views and owned storage, backed by `leapp-triton`.

use crate::{DataType, RuntimeError};
pub use leapp_triton::MemoryLocation;

mod sealed {
    pub trait Element: leapp_triton::TensorElement {}

    impl<T: leapp_triton::TensorElement> Element for T {}
}

/// Primitive numeric elements accepted by typed tensor helpers. This trait is sealed.
/// Use packed bytes for Boolean, float16, and bfloat16 values.
pub trait TensorElement: sealed::Element {}

impl<T: leapp_triton::TensorElement> TensorElement for T {}

/// Contiguous immutable input borrowed for a synchronous call.
#[derive(Clone, Copy)]
pub struct TensorView<'a>(pub(crate) leapp_triton::TensorView<'a>);

impl<'a> TensorView<'a> {
    /// Borrow CPU values; the concrete shape must match the element count.
    pub fn cpu<T: TensorElement>(data: &'a [T], shape: &'a [i64]) -> Result<Self, RuntimeError> {
        Ok(Self(leapp_triton::TensorView::cpu(data, shape)?))
    }

    /// Borrow packed native-endian numeric bytes. Boolean values must be 0 or 1.
    pub fn cpu_bytes(
        data: &'a [u8],
        dtype: DataType,
        shape: &'a [i64],
    ) -> Result<Self, RuntimeError> {
        Ok(Self(leapp_triton::TensorView::cpu_bytes(
            data,
            dtype.into(),
            shape,
        )?))
    }
}

/// Immutable owned tensor. Clones and views share storage without copying.
/// Outputs remain usable after their result, session, or runtime is dropped.
#[derive(Clone)]
pub struct Tensor(pub(crate) leapp_triton::Tensor);

impl Tensor {
    /// Copy CPU values into ready-to-use storage at the requested location.
    pub fn copy_from_cpu<T: TensorElement>(
        data: &[T],
        shape: &[i64],
        location: MemoryLocation,
    ) -> Result<Self, RuntimeError> {
        Ok(Self(leapp_triton::Tensor::copy_from_cpu(
            data, shape, location,
        )?))
    }

    /// Copy packed native-endian numeric data, including Boolean and half types.
    pub fn copy_from_cpu_bytes(
        data: &[u8],
        dtype: DataType,
        shape: &[i64],
        location: MemoryLocation,
    ) -> Result<Self, RuntimeError> {
        Ok(Self(leapp_triton::Tensor::copy_from_cpu_bytes(
            data,
            dtype.into(),
            shape,
            location,
        )?))
    }

    /// Borrow this tensor as an input without transferring its storage.
    pub fn view(&self) -> TensorView<'_> {
        TensorView(self.0.view())
    }

    /// Scalar type using the same enum as bundle inspection.
    pub fn dtype(&self) -> DataType {
        self.0.dtype().into()
    }

    /// Concrete dimensions; an empty shape denotes a scalar.
    pub fn shape(&self) -> &[i64] {
        self.0.shape()
    }

    /// Location of the owned allocation.
    pub fn location(&self) -> MemoryLocation {
        self.0.location()
    }

    /// Borrow typed host data, rejecting CUDA storage or a mismatched type.
    pub fn as_cpu_slice<T: TensorElement>(&self) -> Result<&[T], RuntimeError> {
        Ok(self.0.as_cpu_slice()?)
    }

    /// Borrow packed host bytes, rejecting CUDA storage.
    pub fn as_cpu_bytes(&self) -> Result<&[u8], RuntimeError> {
        Ok(self.0.as_cpu_bytes()?)
    }

    /// Copy into a typed host vector, waiting for any CUDA transfer to complete.
    pub fn copy_to_cpu<T: TensorElement>(&self) -> Result<Vec<T>, RuntimeError> {
        Ok(self.0.copy_to_cpu()?)
    }

    /// Copy packed bytes to the host, waiting for any CUDA transfer to complete.
    pub fn copy_to_cpu_bytes(&self) -> Result<Vec<u8>, RuntimeError> {
        Ok(self.0.copy_to_cpu_bytes()?)
    }
}

macro_rules! data_types {
    ($($variant:ident),* $(,)?) => {
        impl From<DataType> for leapp_triton::DataType {
            fn from(value: DataType) -> Self {
                match value {
                    $(DataType::$variant => Self::$variant,)*
                }
            }
        }

        impl From<leapp_triton::DataType> for DataType {
            fn from(value: leapp_triton::DataType) -> Self {
                match value {
                    $(leapp_triton::DataType::$variant => Self::$variant,)*
                }
            }
        }
    };
}

data_types!(
    Bool, Uint8, Uint16, Uint32, Uint64, Int8, Int16, Int32, Int64, Float16, Float32, Float64,
    Bfloat16
);

// Native interop adapters retain the validated view's original lifetime.
impl<'a> From<leapp_triton::TensorView<'a>> for TensorView<'a> {
    fn from(value: leapp_triton::TensorView<'a>) -> Self {
        Self(value)
    }
}

impl Tensor {
    /// Native read-only tensor handle for language binding adapters.
    #[doc(hidden)]
    pub fn native(&self) -> &leapp_triton::Tensor {
        &self.0
    }
}
