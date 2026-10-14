// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "leapp/bundle.h"

int main() {
  leapp::Bundle bundle;
  const auto status = leapp::Bundle::Open("", &bundle);
  return !status.ok() && !bundle.is_valid() &&
                 status.code() == leapp::ErrorCode::kInvalidArgument
             ? 0
             : 1;
}
