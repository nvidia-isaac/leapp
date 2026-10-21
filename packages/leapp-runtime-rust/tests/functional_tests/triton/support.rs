// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

use leapp_runtime::{dlpack::*, MemoryLocation};

pub fn placed<T: CpuElement>(
    values: &[T],
    shape: &[i64],
    location: MemoryLocation,
) -> crate::engines::triton::Result<SharedTensor> {
    crate::engines::triton::memory::dlpack::copy(
        BorrowedTensor::cpu(values, shape).unwrap(),
        Some(location),
    )
}

// These fixtures retain their storage and never hand pointers to foreign writers.
// Metadata inspection must not expose outputs that later checks copy or reuse.
pub trait TestTensor {
    fn view(&self) -> BorrowedTensor<'_>;
    fn shape(&self) -> &[i64];
    fn location(&self) -> MemoryLocation;
}

impl TestTensor for SharedTensor {
    fn view(&self) -> BorrowedTensor<'_> {
        // SAFETY: fixture storage remains ready and immutable throughout inference.
        unsafe { self.borrow() }
    }

    fn shape(&self) -> &[i64] {
        let view = self.view();
        let tensor = view.descriptor();
        // SAFETY: inference outputs own valid shape metadata until self is dropped.
        unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) }
    }

    fn location(&self) -> MemoryLocation {
        crate::engines::triton::memory::dlpack::location(self.view().descriptor().device).unwrap()
    }
}
