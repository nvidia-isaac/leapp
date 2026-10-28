// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::bridge;
use crate::{bundle::schema::NodeSpec, MemoryLocation};
use anyhow::Result;

pub(super) fn load(spec: &NodeSpec, location: MemoryLocation) -> Result<bridge::Model> {
    bridge::Model::load(
        spec,
        location,
        "nativert",
        "libleapp_nativert_backend.so",
        true,
    )
}
