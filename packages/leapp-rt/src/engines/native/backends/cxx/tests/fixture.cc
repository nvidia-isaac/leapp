// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "leapp/native/export.h"

namespace fixture {

namespace native = lrt::native;

struct Model {
  std::string mode;
  size_t outputs;
};

// One identity fixture exercises both execution paths and boundary failures.
class Plugin final : public native::BackendPlugin {
 public:
  explicit Plugin(std::shared_ptr<const Model> model)
      : model_(std::move(model)) {}

  lrt::Status new_session(
      std::unique_ptr<native::BackendPlugin>* peer) const override {
    *peer = std::make_unique<Plugin>(model_);
    return {};
  }

  bool can_prepare() const noexcept override {
    return model_->mode == "prepared";
  }

  bool is_prepared() const noexcept override { return prepared_; }

  lrt::Status prepare(
      std::span<const native::BorrowedTensor> inputs,
      std::span<const native::WritableTensor> outputs) override {
    input_ = inputs[0].descriptor();
    output_ = outputs[0].descriptor();
    // Descriptor metadata is temporary; prepared storage stays borrowed.
    if (!inputs[0].shape().empty()) {
      shape_.assign(inputs[0].shape().begin(), inputs[0].shape().end());
    }
    input_.shape = output_.shape = shape_.data();
    prepared_ = true;
    return {};
  }

  lrt::Status run(std::span<const native::NamedTensor> inputs,
                  std::span<const native::InputRoute> routes,
                  std::span<const native::OutputBinding> outputs) override {
    const auto source = routes.empty() ? native::BorrowedTensor(input_)
                                       : inputs[routes[0].boundary].tensor;
    const auto target =
        outputs.empty() ? native::WritableTensor(output_) : outputs[0].tensor;
    if (source.bytes()) {
      std::memcpy(target.data(), source.data(), source.bytes());
    }
    return {};
  }

  lrt::Status run_dynamic(std::span<const native::BorrowedTensor> inputs,
                          std::span<const native::OutputBinding> outputs,
                          native::Transfers& transfers,
                          native::DynamicOutputs* result) override {
    if (model_->mode == "throw") {
      throw std::runtime_error("SDK exception");
    }
    auto descriptor = inputs[0].descriptor();
    if (model_->mode == "copy_error") {
      descriptor.dtype.lanes = 2;
    }
    native::DynamicOutputs candidate(model_->outputs);
    for (size_t i = 0; i < candidate.size(); ++i) {
      const auto target = std::find_if(
          outputs.begin(), outputs.end(),
          [i](const auto& output) { return output.position == i; });
      if (target == outputs.end()) {
        native::OwnedTensor value;
        auto status =
            transfers.copy(native::BorrowedTensor(descriptor), &value);
        if (!status.ok()) {
          return status;
        }
        candidate[i] = std::move(value);
      } else if (inputs[0].bytes()) {
        std::memcpy(target->tensor.data(), inputs[0].data(), inputs[0].bytes());
      }
    }
    if (model_->mode == "partial_error") {
      return {lrt::ErrorCode::kInternal, "SDK", "failed after copying"};
    }
    *result = std::move(candidate);
    return {};
  }

 private:
  std::shared_ptr<const Model> model_;
  std::vector<int64_t> shape_;
  DLTensor input_{}, output_{};
  bool prepared_ = false;
};

lrt::Status load(const native::LoadContext& context,
                 std::unique_ptr<native::BackendPlugin>* output) {
  const std::string mode(context.model_path);
  if (mode.size() == 1 && mode[0] >= '1' && mode[0] <= '5') {
    return {static_cast<lrt::ErrorCode>(mode[0] - '0'), "SDK", "diagnostic"};
  }
  auto model =
      std::make_shared<const Model>(Model{mode, context.outputs.size()});
  *output = std::make_unique<Plugin>(std::move(model));
  return {};
}

}  // namespace fixture

LEAPP_NATIVE_BACKEND(fixture, fixture::load)

extern "C" __attribute__((visibility("default")))
const lrt::native::detail::Header*
leapp_native_incompatible_v1() noexcept {
  static const lrt::native::detail::Header header{2, sizeof(header)};
  return &header;
}
