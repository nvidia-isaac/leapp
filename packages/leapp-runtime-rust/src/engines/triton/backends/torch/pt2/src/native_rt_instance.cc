// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/native_rt_instance.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ATen/Functions.h"
#include "c10/cuda/CUDAGuard.h"
#include "caffe2/serialize/inline_container.h"
#include "nlohmann/json.hpp"

namespace leapp::internal {
namespace {

// ModelRunner preserves archive placement. Check it before allocating weights.
Status CheckArchive(const NativeRtModel& model, c10::Device device) {
  caffe2::serialize::PyTorchStreamReader reader(model.artifact_path.string());
  const auto [data, size] = reader.getRecord("models/model.json");
  const auto archive =
      nlohmann::json::parse(static_cast<const char*>(data.get()),
                            static_cast<const char*>(data.get()) + size);
  const auto& module = archive.at("graph_module");
  const auto& tensors = module.at("graph").at("tensor_values");
  // Include weights and intermediates, including graphs without user inputs.
  for (const auto& metadata : tensors) {
    const auto& recorded = metadata.at("device");
    const auto type = recorded.at("type").get<std::string>();
    const int index =
        recorded.at("index").is_null() ? 0 : recorded.at("index").get<int>();
    if (type != "cpu" &&
        !(device.is_cuda() && type == "cuda" && index == device.index())) {
      return {ErrorCode::kUnsupported, "nativert",
              "PT2 archive devices are incompatible with " + device.str()};
    }
  }
  size_t input_count = 0;
  size_t output_count = 0;
  for (const auto& spec : module.at("signature").at("input_specs")) {
    if (!spec.contains("user_input")) {
      continue;
    }
    const auto& arg = spec.at("user_input").at("arg");
    if (!arg.contains("as_tensor")) {
      return {ErrorCode::kUnsupported, "nativert",
              "PT2 user inputs must be tensors"};
    }
    const auto& metadata =
        tensors.at(arg.at("as_tensor").at("name").get<std::string>());
    const auto& recorded = metadata.at("device");
    const auto type = recorded.at("type").get<std::string>();
    const int index =
        recorded.at("index").is_null() ? 0 : recorded.at("index").get<int>();
    if ((device.is_cpu() && type != "cpu") ||
        (device.is_cuda() && (type != "cuda" || index != device.index()))) {
      return {ErrorCode::kUnsupported, "nativert",
              "PT2 input devices must match the model instance; export for " +
                  device.str() + " (NativeRT does not remap archive devices)"};
    }
    ++input_count;
  }
  for (const auto& spec : module.at("signature").at("output_specs")) {
    if (spec.contains("user_output")) {
      if (!spec.at("user_output").at("arg").contains("as_tensor")) {
        return {ErrorCode::kUnsupported, "nativert",
                "PT2 user outputs must be tensors"};
      }
      ++output_count;
    }
  }
  if (input_count != static_cast<size_t>(model.inputs.size()) ||
      output_count != static_cast<size_t>(model.outputs.size())) {
    return {ErrorCode::kInvalidArgument, "nativert",
            "PT2 tensor counts do not match the model configuration"};
  }
  return {};
}

}  // namespace

Status NativeRtInstance::Create(TRITONBACKEND_ModelInstance* instance,
                                std::unique_ptr<NativeRtInstance>* result) {
  TRITONBACKEND_Model* model = nullptr;
  void* state = nullptr;
  auto status =
      TakeBackendError(TRITONBACKEND_ModelInstanceModel(instance, &model));
  if (!status.ok()) {
    return status;
  }
  status = TakeBackendError(TRITONBACKEND_ModelState(model, &state));
  if (!status.ok()) {
    return status;
  }
  auto value = std::unique_ptr<NativeRtInstance>(
      new NativeRtInstance(*static_cast<NativeRtModel*>(state)));
  status = InstanceDevice(instance, &value->device_, &value->stream_);
  if (!status.ok()) {
    return status;
  }
  const auto& stream = value->stream_;
  std::optional<c10::cuda::CUDAStreamGuard> guard;
  if (stream) {
    guard.emplace(*stream);
  }
  status = CheckArchive(value->model_, value->device_);
  if (!status.ok()) {
    return status;
  }
  for (const auto& port : value->model_.inputs) {
    if (TorchType(port.dtype) == at::ScalarType::Undefined) {
      return {ErrorCode::kUnsupported, "nativert",
              "unsupported PT2 input dtype"};
    }
  }
  value->runner_ = std::make_unique<torch::nativert::ModelRunnerHandle>(
      value->model_.artifact_path.string(), "model");
  if (stream) {
    stream->synchronize();
  }
  *result = std::move(value);
  return {};
}

Status NativeRtInstance::Execute(TRITONBACKEND_Request* request,
                                 TRITONBACKEND_Response* response) {
  std::optional<c10::cuda::CUDAStreamGuard> guard;
  if (stream_) {
    guard.emplace(*stream_);
  }
  std::vector<c10::IValue> inputs;
  std::vector<c10::IValue> outputs;
  // Keep owners and the selected stream alive through failure cleanup as well.
  try {
    const std::vector<InputAccess> access(model_.inputs.size(),
                                          InputAccess::kRequireOwnedStorage);
    auto status = ReadInputs(request, model_.inputs, access, device_, &inputs);
    if (status.ok()) {
      outputs = runner_->runWithFlatInputsAndOutputs(std::move(inputs));
      status =
          WriteOutputs(request, model_.outputs, outputs, response, stream_);
    }
    if (stream_) {
      stream_->synchronize();
    }
    return status;
  } catch (const std::exception& error) {
    if (stream_) {
      stream_->synchronize();
    }
    return {ErrorCode::kInternal, "nativert", error.what()};
  } catch (...) {
    if (stream_) {
      stream_->synchronize();
    }
    return {ErrorCode::kInternal, "nativert", "unknown NativeRT exception"};
  }
}

}  // namespace leapp::internal
