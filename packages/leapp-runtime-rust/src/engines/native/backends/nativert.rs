// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::bridge;
use crate::{bundle::schema::NodeSpec, MemoryLocation};
use anyhow::Result;

pub(super) fn load(
    spec: &NodeSpec,
    location: MemoryLocation,
    stream: Option<crate::CudaStream>,
) -> Result<bridge::Model> {
    bridge::Model::load(
        spec,
        location,
        stream,
        "nativert",
        "libleapp_nativert_backend.so",
        true,
    )
}
