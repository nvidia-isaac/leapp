// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "leapp/bundle.h"

int main() {
  lrt::Bundle bundle;
  const auto status = lrt::Bundle::Open("", &bundle);
  return !status.ok() && !bundle.is_valid() &&
                 bundle.model_io("absent") == nullptr &&
                 status.code() == lrt::ErrorCode::kInvalidArgument
             ? 0
             : 1;
}
