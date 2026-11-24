// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "leapp/native/export.h"

lrt::Status Load(const lrt::native::LoadContext&,
                 std::unique_ptr<lrt::native::BackendPlugin>*) {
  return {lrt::ErrorCode::kUnsupported, "load", "test backend"};
}

LEAPP_NATIVE_BACKEND(example, Load)
