#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared schema helpers for ``leapp::warp_runner``."""

from __future__ import annotations

import torch

NAMESPACE = "leapp"
OP_NAME = "warp_runner"
QUALIFIED_NAME = f"{NAMESPACE}::{OP_NAME}"

_SCHEMA = f"{OP_NAME}(Tensor[] inputs, str runtime_metadata, Tensor bundle) -> Tensor[]"

ONNX_WRP_DOMAIN = "com.nvidia.warp"
ONNX_WRP_OPSET = 1
ONNX_WRP_OP_TYPE = "WrpRunner"


def get_op() -> "torch._ops.OpOverloadPacket":
    return getattr(getattr(torch.ops, NAMESPACE), OP_NAME)


def _resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{QUALIFIED_NAME}: unknown output dtype name '{name}'")
    return dtype
