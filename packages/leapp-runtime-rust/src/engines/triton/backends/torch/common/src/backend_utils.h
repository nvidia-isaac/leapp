// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef COMMON_SRC_BACKEND_UTILS_H_
#define COMMON_SRC_BACKEND_UTILS_H_

#include <cstdint>
#include <optional>

#include "c10/core/Device.h"
#include "c10/cuda/CUDAStream.h"
#include "leapp/status.h"
#include "triton/core/tritonbackend.h"

namespace leapp::internal {

Status TakeBackendError(TRITONSERVER_Error* error);

TRITONSERVER_Error* NewBackendError(const Status& status);

Status BufferDevice(TRITONSERVER_MemoryType type, int64_t id,
                    c10::Device* device);

Status InstanceDevice(TRITONBACKEND_ModelInstance* instance,
                      c10::Device* device,
                      std::optional<c10::cuda::CUDAStream>* stream);

using ExecuteRequest = Status (*)(void*, TRITONBACKEND_Request*,
                                  TRITONBACKEND_Response*);

TRITONSERVER_Error* ExecuteRequests(void* state, ExecuteRequest execute,
                                    TRITONBACKEND_Request** requests,
                                    uint32_t request_count);

}  // namespace leapp::internal

#endif  // COMMON_SRC_BACKEND_UTILS_H_
