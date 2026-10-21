// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/backend_utils.h"

#include <exception>
#include <limits>
#include <memory>
#include <vector>

namespace leapp::internal {

Status TakeBackendError(TRITONSERVER_Error* error) {
  const std::unique_ptr<TRITONSERVER_Error, decltype(&TRITONSERVER_ErrorDelete)>
      owner(error, TRITONSERVER_ErrorDelete);
  if (error == nullptr) {
    return {};
  }
  return {ErrorCode::kInternal, "backend", TRITONSERVER_ErrorMessage(error)};
}

TRITONSERVER_Error* NewBackendError(const Status& status) {
  if (status.ok()) {
    return nullptr;
  }
  const auto code = status.code() == ErrorCode::kUnsupported
                        ? TRITONSERVER_ERROR_UNSUPPORTED
                    : status.code() == ErrorCode::kInvalidArgument
                        ? TRITONSERVER_ERROR_INVALID_ARG
                        : TRITONSERVER_ERROR_INTERNAL;
  return TRITONSERVER_ErrorNew(code, status.message().c_str());
}

Status BufferDevice(TRITONSERVER_MemoryType type, int64_t id,
                    c10::Device* device) {
  if (type == TRITONSERVER_MEMORY_CPU ||
      type == TRITONSERVER_MEMORY_CPU_PINNED) {
    *device = c10::Device(c10::kCPU);
    return {};
  }
  if (type == TRITONSERVER_MEMORY_GPU && id >= 0 &&
      id <= std::numeric_limits<c10::DeviceIndex>::max()) {
    *device = c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(id));
    return {};
  }
  return {ErrorCode::kInvalidArgument, "backend", "invalid buffer device"};
}

Status InstanceDevice(TRITONBACKEND_ModelInstance* instance,
                      c10::Device* device,
                      std::optional<c10::cuda::CUDAStream>* stream) {
  TRITONSERVER_InstanceGroupKind kind{};
  auto status =
      TakeBackendError(TRITONBACKEND_ModelInstanceKind(instance, &kind));
  if (!status.ok()) {
    return status;
  }
  if (kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    int32_t id = 0;
    status =
        TakeBackendError(TRITONBACKEND_ModelInstanceDeviceId(instance, &id));
    if (!status.ok()) {
      return status;
    }
    status = BufferDevice(TRITONSERVER_MEMORY_GPU, id, device);
    if (!status.ok()) {
      return status;
    }
    *stream = c10::cuda::getStreamFromPool(false, device->index());
  } else if (kind != TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    return {ErrorCode::kUnsupported, "backend",
            "instances require KIND_CPU or KIND_GPU"};
  }
  return {};
}

TRITONSERVER_Error* ExecuteRequests(void* state, ExecuteRequest execute,
                                    TRITONBACKEND_Request** requests,
                                    uint32_t request_count) {
  using ResponseOwner =
      std::unique_ptr<TRITONBACKEND_Response,
                      decltype(&TRITONBACKEND_ResponseDelete)>;
  std::vector<ResponseOwner> responses;
  // Allocation failures must leave the entire request batch with Triton.
  try {
    responses.reserve(request_count);
    for (uint32_t i = 0; i < request_count; ++i) {
      TRITONBACKEND_Response* response = nullptr;
      if (auto* error = TRITONBACKEND_ResponseNew(&response, requests[i])) {
        return error;
      }
      responses.emplace_back(response, TRITONBACKEND_ResponseDelete);
    }
  } catch (const std::exception& error) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error.what());
  } catch (...) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                 "unknown exception while creating responses");
  }
  for (uint32_t i = 0; i < request_count; ++i) {
    TRITONSERVER_Error* error = nullptr;
    try {
      error = leapp::internal::NewBackendError(
          execute(state, requests[i], responses[i].get()));
    } catch (const std::exception& exception) {
      error =
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, exception.what());
    } catch (...) {
      error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                    "unknown exception during model execution");
    }
    TRITONSERVER_ErrorDelete(TRITONBACKEND_ResponseSend(
        responses[i].release(), TRITONSERVER_RESPONSE_COMPLETE_FINAL, error));
    TRITONSERVER_ErrorDelete(error);
    TRITONSERVER_ErrorDelete(TRITONBACKEND_RequestRelease(
        requests[i], TRITONSERVER_REQUEST_RELEASE_ALL));
  }
  return nullptr;
}

}  // namespace leapp::internal
