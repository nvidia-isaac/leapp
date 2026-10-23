// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef JIT_SRC_TORCHSCRIPT_INSTANCE_H_
#define JIT_SRC_TORCHSCRIPT_INSTANCE_H_

#include <memory>
#include <optional>

#include "src/backend_utils.h"
#include "src/input_analysis.h"
#include "src/model_config.h"

namespace leapp::internal {

class TorchScriptInstance {
 public:
  static Status Create(TRITONBACKEND_ModelInstance* instance,
                       std::unique_ptr<TorchScriptInstance>* result);

  Status Execute(TRITONBACKEND_Request* request,
                 TRITONBACKEND_Response* response);

 private:
  explicit TorchScriptInstance(const ModelConfig& model) : model_(model) {}

  const ModelConfig& model_;
  c10::Device device_{c10::kCPU};
  std::optional<c10::cuda::CUDAStream> stream_;
  torch::jit::Module module_;
  std::optional<torch::jit::Method> method_;
  InputPlan plan_;
};

}  // namespace leapp::internal

#endif  // JIT_SRC_TORCHSCRIPT_INSTANCE_H_
