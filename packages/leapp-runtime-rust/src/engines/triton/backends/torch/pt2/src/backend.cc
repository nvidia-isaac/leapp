// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "src/native_rt_instance.h"

extern "C" {

TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend*) {
  uint32_t major = 0;
  uint32_t minor = 0;
  if (auto* error = TRITONBACKEND_ApiVersion(&major, &minor)) {
    return error;
  }
  if (major != TRITONBACKEND_API_VERSION_MAJOR ||
      minor < TRITONBACKEND_API_VERSION_MINOR) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                                 "incompatible Triton backend API");
  }
  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  try {
    auto state = std::make_unique<leapp::internal::NativeRtModel>();
    auto status = leapp::internal::ReadModelConfig(model, state.get());
    if (!status.ok()) {
      return leapp::internal::NewBackendError(status);
    }
    auto* transferred = state.release();
    if (auto* error = TRITONBACKEND_ModelSetState(model, transferred)) {
      state.reset(transferred);
      return error;
    }
    return nullptr;
  } catch (const std::exception& error) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error.what());
  } catch (...) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                 "unknown exception while loading PT2 model");
  }
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  void* state = nullptr;
  if (auto* error = TRITONBACKEND_ModelState(model, &state)) {
    return error;
  }
  delete static_cast<leapp::internal::NativeRtModel*>(state);
  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  try {
    std::unique_ptr<leapp::internal::NativeRtInstance> state;
    const auto status =
        leapp::internal::NativeRtInstance::Create(instance, &state);
    if (!status.ok()) {
      return leapp::internal::NewBackendError(status);
    }
    auto* transferred = state.release();
    if (auto* error =
            TRITONBACKEND_ModelInstanceSetState(instance, transferred)) {
      state.reset(transferred);
      return error;
    }
    return nullptr;
  } catch (const std::exception& error) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error.what());
  } catch (...) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "unknown exception while creating PT2 instance");
  }
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  void* state = nullptr;
  if (auto* error = TRITONBACKEND_ModelInstanceState(instance, &state)) {
    return error;
  }
  delete static_cast<leapp::internal::NativeRtInstance*>(state);
  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
  void* state = nullptr;
  if (auto* error = TRITONBACKEND_ModelInstanceState(instance, &state)) {
    return error;
  }
  return leapp::internal::ExecuteRequests(
      state,
      [](void* state, TRITONBACKEND_Request* request,
         TRITONBACKEND_Response* response) {
        return static_cast<leapp::internal::NativeRtInstance*>(state)->Execute(
            request, response);
      },
      requests, request_count);
}

}  // extern "C"
