// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-independent device and memory types, re-exported at the crate root.

/// Device used by model instances, independently of input and output storage.
#[derive(Debug, Clone, Copy, Default)]
pub enum ExecutionDevice {
    #[default]
    Cpu,
    Cuda {
        device_id: u32,
    },
}

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
