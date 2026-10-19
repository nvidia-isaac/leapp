// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/torchscript_instance.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "c10/core/InferenceMode.h"
#include "c10/cuda/CUDAGuard.h"
#include "torch/csrc/jit/serialization/import.h"

namespace leapp::internal {
namespace {

Status ValidateOutputs(const torch::jit::Module& module, size_t expected) {
  const auto& returns =
      module.get_method("forward").function().getSchema().returns();
  if (returns.size() == 1) {
    const auto& type = returns.front().type();
    if (type->kind() == c10::TypeKind::TensorType && expected == 1) {
      return {};
    }
    if (const auto tuple = type->cast<c10::TupleType>()) {
      if (tuple->elements().size() == expected &&
          std::all_of(tuple->elements().begin(), tuple->elements().end(),
                      [](const auto& element) {
                        return element->kind() == c10::TypeKind::TensorType;
                      })) {
        return {};
      }
    }
  }
  return {ErrorCode::kUnsupported, "torchscript",
          "forward must return the configured number of tensors, as a tensor "
          "or a flat tuple"};
}

}  // namespace

Status TorchScriptInstance::Create(
    TRITONBACKEND_ModelInstance* instance,
    std::unique_ptr<TorchScriptInstance>* result) {
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
  auto value = std::unique_ptr<TorchScriptInstance>(
      new TorchScriptInstance(*static_cast<ModelConfig*>(state)));
  status = InstanceDevice(instance, &value->device_, &value->stream_);
  if (!status.ok()) {
    return status;
  }
  std::optional<c10::cuda::CUDAStreamGuard> guard;
  if (value->stream_) {
    guard.emplace(*value->stream_);
  }
  try {
    value->module_ =
        torch::jit::load(value->model_.artifact_path.string(), value->device_);
    value->module_.eval();
    status = AnalyzeInputs(value->module_, &value->plan_);
    if (status.ok() &&
        value->plan_.access.size() != value->model_.inputs.size()) {
      status = {ErrorCode::kInvalidArgument, "torchscript",
                "forward argument count does not match model inputs"};
    }
    if (status.ok()) {
      status = ValidateOutputs(value->module_, value->model_.outputs.size());
      value->method_.emplace(value->module_.get_method("forward"));
    }
    if (value->stream_) {
      value->stream_->synchronize();
    }
  } catch (...) {
    if (value->stream_) {
      value->stream_->synchronize();
    }
    throw;
  }
  if (!status.ok()) {
    return status;
  }
  *result = std::move(value);
  return {};
}

Status TorchScriptInstance::Execute(TRITONBACKEND_Request* request,
                                    TRITONBACKEND_Response* response) {
  std::optional<c10::cuda::CUDAStreamGuard> guard;
  if (stream_) {
    guard.emplace(*stream_);
  }
  c10::InferenceMode inference;
  std::vector<c10::IValue> inputs;
  torch::jit::Stack stack;
  // Owners outlive stream completion on both success and failure.
  try {
    auto status =
        ReadInputs(request, model_.inputs, plan_.access, device_, &inputs);
    if (status.ok()) {
      // Signature and positional tensors were checked at load and import.
      stack.reserve(inputs.size() + 1);
      stack.emplace_back(method_->raw_owner());
      stack.insert(stack.end(), inputs.begin(), inputs.end());
      method_->function().run(stack);
      TORCH_CHECK(stack.size() == 1, "expected one TorchScript return value");
      const auto& output = stack.front();
      std::span<const c10::IValue> outputs;
      if (output.isTensor()) {
        outputs = {&output, 1};
      } else if (output.isTuple()) {
        const auto& elements = output.toTupleRef().elements();
        outputs = {elements.begin(), elements.size()};
      } else {
        status = {ErrorCode::kUnsupported, "torchscript",
                  "forward must return a tensor or a flat tuple of tensors"};
      }
      if (status.ok()) {
        status =
            WriteOutputs(request, model_.outputs, outputs, response, stream_);
      }
    }
    if (stream_) {
      stream_->synchronize();
    }
    return status;
  } catch (const std::exception& error) {
    if (stream_) {
      stream_->synchronize();
    }
    return {ErrorCode::kInternal, "torchscript", error.what()};
  } catch (...) {
    if (stream_) {
      stream_->synchronize();
    }
    return {ErrorCode::kInternal, "torchscript",
            "unknown TorchScript exception"};
  }
}

}  // namespace leapp::internal
