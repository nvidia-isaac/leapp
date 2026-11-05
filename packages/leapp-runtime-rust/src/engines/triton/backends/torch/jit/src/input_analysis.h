// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef JIT_SRC_INPUT_ANALYSIS_H_
#define JIT_SRC_INPUT_ANALYSIS_H_

#include <vector>

#include "src/tensor_io.h"
#include "torch/csrc/jit/api/module.h"

namespace leapp::internal {

struct InputPlan {
  std::vector<InputAccess> access;
};

Status AnalyzeInputs(const torch::jit::Module& module, InputPlan* plan);

}  // namespace leapp::internal

#endif  // JIT_SRC_INPUT_ANALYSIS_H_
