// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef COMMON_SRC_MODEL_CONFIG_H_
#define COMMON_SRC_MODEL_CONFIG_H_

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "leapp/status.h"
#include "triton/core/tritonbackend.h"

namespace leapp::internal {

struct ModelPort {
  std::string name;
  TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INVALID;
  std::vector<int64_t> shape;
};

struct ModelConfig {
  std::filesystem::path artifact_path;
  std::vector<ModelPort> inputs;
  std::vector<ModelPort> outputs;
};

Status ReadModelConfig(TRITONBACKEND_Model* model, ModelConfig* config,
                       std::filesystem::path* custom_op_library = nullptr);

}  // namespace leapp::internal

#endif  // COMMON_SRC_MODEL_CONFIG_H_
