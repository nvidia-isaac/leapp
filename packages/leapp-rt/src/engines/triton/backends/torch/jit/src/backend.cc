// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "src/torchscript_instance.h"

namespace leapp::internal {

namespace {

Status ReadModel(TRITONBACKEND_Model* model, ModelConfig* state) {
  auto status = ReadModelConfig(model, state);
  if (!status.ok()) {
    return status;
  }
  // Ordinals, rather than JSON/config ordering, define forward argument order.
  const auto order = [](std::vector<ModelPort>* ports, const char* prefix) {
    std::vector<ModelPort> ordered;
    for (size_t i = 0; i < ports->size(); ++i) {
      const auto name = std::string(prefix) + std::to_string(i);
      const auto found =
          std::find_if(ports->begin(), ports->end(),
                       [&](const auto& port) { return port.name == name; });
      if (found == ports->end()) {
        return false;
      }
      ordered.push_back(*found);
    }
    *ports = std::move(ordered);
    return true;
  };
  if (!order(&state->inputs, "INPUT__") ||
      !order(&state->outputs, "OUTPUT__")) {
    return {ErrorCode::kInvalidArgument, "torchscript",
            "model ports must use consecutive INPUT__/OUTPUT__ ordinals"};
  }
  for (const auto& port : state->inputs) {
    if (TorchType(port.dtype) == at::ScalarType::Undefined) {
      return {ErrorCode::kUnsupported, "torchscript",
              "unsupported input dtype"};
    }
  }
  return {};
}

}  // namespace
}  // namespace leapp::internal

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
    auto state = std::make_unique<leapp::internal::ModelConfig>();
    auto status = leapp::internal::ReadModel(model, state.get());
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
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "unknown exception while loading TorchScript model");
  }
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  void* state = nullptr;
  if (auto* error = TRITONBACKEND_ModelState(model, &state)) {
    return error;
  }
  delete static_cast<leapp::internal::ModelConfig*>(state);
  return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  try {
    std::unique_ptr<leapp::internal::TorchScriptInstance> state;
    const auto status =
        leapp::internal::TorchScriptInstance::Create(instance, &state);
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
        "unknown exception while creating TorchScript instance");
  }
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  void* state = nullptr;
  if (auto* error = TRITONBACKEND_ModelInstanceState(instance, &state)) {
    return error;
  }
  delete static_cast<leapp::internal::TorchScriptInstance*>(state);
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
        return static_cast<leapp::internal::TorchScriptInstance*>(state)
            ->Execute(request, response);
      },
      requests, request_count);
}

}  // extern "C"
