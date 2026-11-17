// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace leapp_torch::native {

struct Port {
  uint8_t dtype;
  std::vector<int64_t> shape;
};

at::ScalarType scalar_type(uint8_t dtype);

enum class InputAccess { Borrow, Scratch, Owned };

struct InputPlan {
  std::vector<InputAccess> access;
  bool background_work = false;
};

class Model {
 public:
  virtual ~Model() = default;

  virtual std::vector<at::Tensor> forward(
      const std::vector<c10::IValue>& args) = 0;

  InputPlan plan;
};

std::unique_ptr<Model> load_model(const std::string& path, at::Device device,
                                  std::span<const Port> inputs,
                                  std::span<const Port> outputs);

}  // namespace leapp_torch::native
