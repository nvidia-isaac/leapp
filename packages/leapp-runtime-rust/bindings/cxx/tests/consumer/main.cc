// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <iostream>
#include <span>

#include "leapp/runtime.h"

int main(int argc, char** argv) {
  if (argc != 2) {
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
  leapp::SessionOptions options;
  leapp::Session session;
  status = leapp::Session::Open(bundle, options, &session);
  if (!status.ok()) {
    return 1;
  }
  const std::array<float, 4> input = {1, 1, 1, 1};
  int64_t shape = 4;
  DLTensor descriptor{const_cast<float*>(input.data()), {kDLCPU, 0}, 1,
                      {kDLFloat, 32, 1}, &shape, nullptr, 0};
  leapp::RunResult result;
  status = session.Run(
      {{"accumulator/observation", &descriptor}},
      &result);
  if (!status.ok()) {
    return 1;
  }
  leapp::ManagedTensor output;
  status = result.GetTensor("accumulator/next", &output);
  if (!status.ok()) {
    return 1;
  }
  std::span<const float> values(static_cast<const float*>(output->dl_tensor.data), 4);
  if (!status.ok() || values.size() != 4 || values[0] != 1) {
    return 1;
  }
  status = session.Close();
  if (!status.ok() || values[0] != 1) {
    return 1;
  }
  std::cout << "Installed SDK consumer passed\n";
}
