// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <iostream>
#include <span>

#include "leapp/runtime.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    return 2;
  }
  leapp::Bundle bundle;
  auto status = leapp::Bundle::Open(argv[1], &bundle);
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }
  const auto* ports = bundle.model_io("accumulator");
  if (ports == nullptr || ports->inputs.empty() || ports->outputs.empty()) {
    return 1;
  }
  leapp::RuntimeOptions options;
  options.backend_directory = argv[2];
  leapp::Runtime runtime;
  status = leapp::Runtime::Open(bundle, options, &runtime);
  if (!status.ok()) {
    return 1;
  }
  leapp::Session session;
  status = runtime.CreateSession(&session);
  if (!status.ok()) {
    return 1;
  }
  const std::array<float, 4> input = {1, 1, 1, 1};
  leapp::RunResult result;
  status = session.Run(
      {{"accumulator/observation", leapp::TensorView::cpu<float>(input, {4})}},
      {}, &result);
  if (!status.ok()) {
    return 1;
  }
  leapp::Tensor output;
  status = result.GetTensor("accumulator/next", &output);
  if (!status.ok()) {
    return 1;
  }
  std::span<const float> values;
  status = output.CpuSpan(&values);
  if (!status.ok() || values.size() != 4 || values[0] != 1) {
    return 1;
  }
  status = runtime.Close();
  if (!status.ok() || values[0] != 1) {
    return 1;
  }
  std::cout << "Installed SDK consumer passed\n";
}
