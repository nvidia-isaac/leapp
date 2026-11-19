// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "leapp/native/export.h"

namespace {
class Plugin final : public lrt::native::BackendPlugin {
 public:
  lrt::Status new_session(
      std::unique_ptr<BackendPlugin>* output) const override {
    *output = std::make_unique<Plugin>();
    return {};
  }
  lrt::Status run_dynamic(std::span<const lrt::native::BorrowedTensor>,
                          std::span<const lrt::native::OutputBinding>,
                          lrt::native::Transfers&,
                          lrt::native::DynamicOutputs* output) override {
    output->clear();
    return {};
  }
};
lrt::Status Load(const lrt::native::LoadContext&,
                 std::unique_ptr<lrt::native::BackendPlugin>* output) {
  *output = std::make_unique<Plugin>();
  return {};
}
}  // namespace

LEAPP_NATIVE_BACKEND(example, Load)
