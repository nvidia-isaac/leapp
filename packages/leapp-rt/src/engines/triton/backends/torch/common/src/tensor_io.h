// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef COMMON_SRC_TENSOR_IO_H_
#define COMMON_SRC_TENSOR_IO_H_

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "ATen/core/ivalue.h"
#include "c10/cuda/CUDAStream.h"
#include "src/model_config.h"

namespace leapp::internal {

enum class InputAccess { kBorrowReadOnly, kRequireOwnedStorage };

struct InputBuffer {
  const void* data;
  uint64_t bytes;
  c10::Device device;
};

at::ScalarType TorchType(TRITONSERVER_DataType type);

Status PrepareInput(at::IntArrayRef shape, at::ScalarType dtype,
                    std::span<const InputBuffer> buffers, InputAccess access,
                    c10::Device device, at::Tensor* tensor);

Status ReadInputs(TRITONBACKEND_Request* request,
                  std::span<const ModelPort> ports,
                  std::span<const InputAccess> access, c10::Device device,
                  std::vector<c10::IValue>* inputs);

// Retain outputs and response buffers until stream completion, also on error.
Status WriteOutputs(TRITONBACKEND_Request* request,
                    std::span<const ModelPort> ports,
                    std::span<const c10::IValue> outputs,
                    TRITONBACKEND_Response* response,
                    const std::optional<c10::cuda::CUDAStream>& stream);

}  // namespace leapp::internal

#endif  // COMMON_SRC_TENSOR_IO_H_
