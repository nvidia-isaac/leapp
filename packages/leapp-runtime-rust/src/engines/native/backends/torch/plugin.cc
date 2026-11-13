// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include "leapp/native/export.h"
#include "torch/executor.h"

namespace leapp_torch::native {

namespace {

namespace adapter = leapp_native::tensors;
namespace plugin = lrt::native;

// Implement the native plugin lifecycle; executor.cc owns SDK execution
// details.
class Plugin final : public lrt::native::BackendPlugin {
 public:
  explicit Plugin(std::unique_ptr<Executor> executor)
      : executor_(std::move(executor)) {}

  lrt::Status new_session(
      std::unique_ptr<plugin::BackendPlugin>* peer) const override {
    *peer = std::make_unique<Plugin>(executor_->new_session());
    return {};
  }

  bool can_prepare() const noexcept override { return true; }

  bool is_prepared() const noexcept override {
    return executor_->is_prepared();
  }

  size_t input_alignment() const noexcept override {
    return executor_->input_alignment();
  }

  lrt::Status prepare(
      std::span<const plugin::BorrowedTensor> inputs,
      std::span<const plugin::WritableTensor> outputs) override {
    inputs_.assign(inputs);
    const auto targets = adapter::views(outputs);
    executor_->prepare(inputs_.values, targets);
    return {};
  }

  lrt::Status run(std::span<const plugin::NamedTensor> inputs,
                  std::span<const plugin::InputRoute> routes,
                  std::span<const plugin::OutputBinding> outputs) override {
    adapter::FixedViews::Invocation borrowed(inputs_, inputs, routes);
    const auto targets = adapter::bindings(outputs);
    executor_->run(inputs_.values, routes, targets);
    return {};
  }

  lrt::Status run_dynamic(std::span<const plugin::BorrowedTensor> inputs,
                          std::span<const plugin::OutputBinding> outputs,
                          plugin::Transfers& transfers,
                          plugin::DynamicOutputs* result) override {
    const auto values = adapter::views(inputs);
    const auto targets = adapter::bindings(outputs);
    auto sdk = executor_->run_dynamic(values, targets);
    plugin::DynamicOutputs candidate(sdk->views.size());
    for (size_t i = 0; i < candidate.size(); ++i) {
      if (std::none_of(outputs.begin(), outputs.end(), [i](const auto& output) {
            return output.position == i;
          })) {
        plugin::OwnedTensor value;
        auto status = transfers.copy(
            plugin::BorrowedTensor(adapter::descriptor(sdk->views[i])), &value);
        if (!status.ok()) {
          return status;
        }
        candidate[i] = std::move(value);
      }
    }
    *result = std::move(candidate);
    return {};
  }

 private:
  adapter::FixedViews inputs_;
  std::unique_ptr<Executor> executor_;
};

lrt::Status load(const plugin::LoadContext& context,
                 std::unique_ptr<plugin::BackendPlugin>* output) {
  const auto inputs = adapter::ports(context.inputs);
  const auto outputs = adapter::ports(context.outputs);
  auto executor = std::make_unique<Executor>(
      std::string(context.model_path), context.device, context.cuda_stream,
      inputs, outputs);
  *output = std::make_unique<Plugin>(std::move(executor));
  return {};
}

}  // namespace

}  // namespace leapp_torch::native

LEAPP_NATIVE_BACKEND(LEAPP_TORCH_BACKEND, leapp_torch::native::load)
