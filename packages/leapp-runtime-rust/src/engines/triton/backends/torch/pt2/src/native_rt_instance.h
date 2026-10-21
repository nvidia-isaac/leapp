// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PT2_SRC_NATIVE_RT_INSTANCE_H_
#define PT2_SRC_NATIVE_RT_INSTANCE_H_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "ATen/core/ivalue.h"
#include "c10/cuda/CUDAStream.h"
#include "leapp/status.h"
#include "src/backend_utils.h"
#include "src/model_config.h"
#include "src/tensor_io.h"
#include "torch/nativert/ModelRunnerHandle.h"
#include "triton/core/tritonbackend.h"

namespace leapp::internal {

struct NativeRtModel : ModelConfig {};

// Triton owns model state until all its instances have been destroyed.
class NativeRtInstance {
 public:
  static Status Create(TRITONBACKEND_ModelInstance* instance,
                       std::unique_ptr<NativeRtInstance>* result);

  Status Execute(TRITONBACKEND_Request* request,
                 TRITONBACKEND_Response* response);

 private:
  explicit NativeRtInstance(const NativeRtModel& model) : model_(model) {}

  const NativeRtModel& model_;
  c10::Device device_{c10::kCPU};
  std::optional<c10::cuda::CUDAStream> stream_;
  std::unique_ptr<torch::nativert::ModelRunnerHandle> runner_;
};

}  // namespace leapp::internal

#endif  // PT2_SRC_NATIVE_RT_INSTANCE_H_
