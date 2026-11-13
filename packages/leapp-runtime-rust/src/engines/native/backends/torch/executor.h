// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <span>
#include <string>

#include "../bridge/tensor_view.h"

namespace leapp_torch::native {

// Shared Torch execution layer for TorchScript and NativeRT. Each peer keeps
// private wrappers, streams, and prepared calls while sharing loaded model
// state.
class Executor {
 public:
  Executor(const std::string& path, int32_t device, void* stream,
           std::span<const leapp_native::tensors::Port> inputs,
           std::span<const leapp_native::tensors::Port> outputs);

  ~Executor();

  std::unique_ptr<Executor> new_session() const;

  bool is_prepared() const noexcept;
  size_t input_alignment() const noexcept;

  void prepare(std::span<const leapp_native::tensors::View> inputs,
               std::span<const leapp_native::tensors::View> outputs);

  void run(std::span<const leapp_native::tensors::View> inputs,
           std::span<const leapp_native::tensors::InputRoute> routes,
           std::span<const leapp_native::tensors::OutputBinding> outputs);

  std::unique_ptr<leapp_native::tensors::Outputs> run_dynamic(
      std::span<const leapp_native::tensors::View> inputs,
      std::span<const leapp_native::tensors::OutputBinding> outputs);

 private:
  struct State;
  explicit Executor(std::unique_ptr<State> state);

  std::unique_ptr<State> state_;
};

}  // namespace leapp_torch::native
