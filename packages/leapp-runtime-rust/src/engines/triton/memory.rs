// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Triton-specific memory tags and tensor conversion.
pub(super) mod dlpack;
use super::native::sys;
use crate::MemoryLocation;

impl MemoryLocation {
    pub(crate) fn native(self) -> (u32, i64) {
        match self {
            Self::Cpu => (sys::TRITONSERVER_MEMORY_CPU, 0),
            Self::PinnedCpu => (sys::TRITONSERVER_MEMORY_CPU_PINNED, 0),
            Self::Cuda { device_id } => (sys::TRITONSERVER_MEMORY_GPU, device_id.into()),
        }
    }
}
