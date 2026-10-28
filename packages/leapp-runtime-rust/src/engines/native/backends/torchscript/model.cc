// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include "input_analysis.h"

namespace leapp_torch::native {

namespace {

class JitModel final : public Model {
 public:
  JitModel(const std::string& path, at::Device device)
      : module_(torch::jit::load(path, device)),
        method_(module_.get_method("forward")) {
    module_.eval();
    plan = analyze_inputs(module_);
  }

  std::vector<at::Tensor> forward(
      const std::vector<c10::IValue>& args) override {
    // The runtime already validates all positional tensor inputs. Use the same
    // executor entry as Python without another method lookup/schema
    // normalization.
    torch::jit::Stack stack;
    stack.reserve(args.size() + 1);
    stack.emplace_back(method_.raw_owner());
    stack.insert(stack.end(), args.begin(), args.end());
    method_.function().run(stack);
    TORCH_CHECK(stack.size() == 1, "expected one TorchScript return value");
    const auto& value = stack.front();
    if (value.isTensor()) {
      return {value.toTensor()};
    }
    std::vector<at::Tensor> result;
    auto append = [&](const c10::IValue& item) {
      TORCH_CHECK(item.isTensor(), "expected flat tensor outputs");
      result.push_back(item.toTensor());
    };
    if (value.isTuple()) {
      for (const auto& item : value.toTuple()->elements()) {
        append(item);
      }
    } else if (value.isList()) {
      for (const auto& item : value.toList()) {
        append(item);
      }
    } else {
      TORCH_CHECK(false, "expected tensor or flat tensor tuple/list output");
    }
    return result;
  }

 private:
  torch::jit::Module module_;
  torch::jit::Method method_;
};

}  // namespace

std::unique_ptr<Model> load_model(const std::string& path, at::Device device,
                                  std::span<const Port>,
                                  std::span<const Port>) {
  return std::make_unique<JitModel>(path, device);
}

}  // namespace leapp_torch::native
