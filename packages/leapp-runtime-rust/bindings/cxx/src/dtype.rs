// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::ffi;
use lrt::DataType;

pub fn wire_dtype(value: DataType) -> ffi::Dtype {
    use DataType as D;
    match value {
        D::Bool => ffi::Dtype::Bool,
        D::Uint8 => ffi::Dtype::Uint8,
        D::Uint16 => ffi::Dtype::Uint16,
        D::Uint32 => ffi::Dtype::Uint32,
        D::Uint64 => ffi::Dtype::Uint64,
        D::Int8 => ffi::Dtype::Int8,
        D::Int16 => ffi::Dtype::Int16,
        D::Int32 => ffi::Dtype::Int32,
        D::Int64 => ffi::Dtype::Int64,
        D::Float16 => ffi::Dtype::Float16,
        D::Bfloat16 => ffi::Dtype::Bfloat16,
        D::Float32 => ffi::Dtype::Float32,
        D::Float64 => ffi::Dtype::Float64,
    }
}
