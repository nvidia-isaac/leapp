// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
    completed_.reset();
    completed_ = executor_->run(inputs_.values, routes, targets);
    return {};
  }

  std::optional<plugin::BorrowedTensor> forwarded_output(
      size_t position) const noexcept override {
    if (!completed_ || position >= completed_->views.size()) {
      return std::nullopt;
    }
    return plugin::BorrowedTensor(
        adapter::descriptor(completed_->views[position]));
  }

  void release_outputs() noexcept override { completed_.reset(); }

  // Lend Torch's results through forwarded_output() instead of copying them;
  // the engine copies only what it publishes.
  lrt::Status run_dynamic(std::span<const plugin::BorrowedTensor> inputs,
                          std::span<const plugin::OutputBinding> outputs,
                          plugin::Transfers&,
                          plugin::DynamicOutputs* result) override {
    const auto values = adapter::views(inputs);
    const auto targets = adapter::bindings(outputs);
    completed_.reset();
    completed_ = executor_->run_dynamic(values, targets);
    *result = plugin::DynamicOutputs(completed_->views.size());
    return {};
  }

 private:
  adapter::FixedViews inputs_;
  std::unique_ptr<Executor> executor_;
  std::unique_ptr<adapter::Outputs> completed_;
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
