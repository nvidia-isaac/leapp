// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Triton-specific memory tags and tensor conversion.
pub(super) mod dlpack;
use super::native::sys;
use crate::MemoryLocation;

impl MemoryLocation {
    pub(crate) fn from_native(memory: u32, device: i64) -> super::Result<Self> {
        match (memory, device) {
            (sys::TRITONSERVER_MEMORY_CPU, 0) => Ok(Self::Cpu),
            (sys::TRITONSERVER_MEMORY_CPU_PINNED, 0) => Ok(Self::PinnedCpu),
            (sys::TRITONSERVER_MEMORY_GPU, device) if u32::try_from(device).is_ok() => {
                Ok(Self::Cuda {
                    device_id: device as u32,
                })
            }
            _ => Err(super::Error::new(
                super::ErrorKind::Internal,
                "execute",
                "unsupported Triton output memory",
            )),
        }
    }

    pub(crate) fn native(self) -> (u32, i64) {
        match self {
            Self::Cpu => (sys::TRITONSERVER_MEMORY_CPU, 0),
            Self::PinnedCpu => (sys::TRITONSERVER_MEMORY_CPU_PINNED, 0),
            Self::Cuda { device_id } => (sys::TRITONSERVER_MEMORY_GPU, device_id.into()),
        }
    }
}
