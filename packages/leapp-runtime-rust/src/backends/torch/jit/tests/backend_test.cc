// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>
#include <source_location>
#include <vector>

#include "ATen/Functions.h"
#include "c10/cuda/CUDAGuard.h"
#include "src/input_analysis.h"
#include "src/tensor_io.h"

namespace {

using leapp::internal::InputAccess;
using leapp::internal::InputBuffer;
using leapp::internal::InputPlan;
using leapp::internal::PrepareInput;

void Check(bool condition,
           std::source_location location = std::source_location::current()) {
  if (!condition) {
    std::cerr << location.file_name() << ":" << location.line()
              << ": backend check failed\n";
    std::abort();
  }
}

void Analysis() {
  for (const auto& source :
       {"def forward(self, x: Tensor):\n    return x + 1\n",
        "def forward(self, x: Tensor):\n    return x.view([-1])\n",
        "def forward(self, x: Tensor):\n    y = x + 1\n    return "
        "y.add_(2)\n"}) {
    torch::jit::Module module("read_only");
    module.define(source);
    InputPlan plan;
    Check(leapp::internal::AnalyzeInputs(module, &plan).ok());
    Check(plan.access == std::vector{InputAccess::kBorrowReadOnly});
  }
  for (const auto& source :
       {"def forward(self, x: Tensor):\n    return x.add_(1)\n",
        "def forward(self, x: Tensor):\n    y = x.view([-1])\n    return "
        "y.add_(1)\n",
        "def forward(self, x: Tensor):\n    if bool(x[0] > 0):\n        "
        "x.add_(1)\n    return x\n",
        "def modify(self, x: Tensor):\n    return x.add_(1)\n"
        "def forward(self, x: Tensor):\n    return self.modify(x)\n"}) {
    torch::jit::Module module("mutating");
    module.define(source);
    InputPlan plan;
    Check(leapp::internal::AnalyzeInputs(module, &plan).ok());
    Check(plan.access == std::vector{InputAccess::kRequireOwnedStorage});
  }
  torch::jit::Module retained("retained");
  retained.register_buffer("saved", at::zeros({4}));
  retained.define(
      "def forward(self, x: Tensor):\n    self.saved = x\n    return x + 1\n");
  InputPlan plan;
  Check(leapp::internal::AnalyzeInputs(retained, &plan).ok());
  Check(plan.access == std::vector{InputAccess::kRequireOwnedStorage});
  torch::jit::Module list_state("list_state");
  list_state.register_attribute("saved", c10::ListType::ofTensors(),
                                c10::List<at::Tensor>());
  list_state.define(
      "def forward(self, x: Tensor):\n    self.saved.append(x)\n    return x + "
      "1\n");
  Check(leapp::internal::AnalyzeInputs(list_state, &plan).ok());
  Check(plan.access == std::vector{InputAccess::kRequireOwnedStorage});
  torch::jit::Module invalid("invalid");
  invalid.define("def forward(self, x: int):\n    return x + 1\n");
  Check(!leapp::internal::AnalyzeInputs(invalid, &plan).ok());
}

void Inputs(c10::Device device) {
  auto original = at::arange(4, at::TensorOptions().dtype(at::kFloat));
  auto source = original.to(device);
  const std::vector<InputBuffer> buffers = {
      {source.data_ptr(), source.nbytes(), device}};
  at::Tensor prepared;
  Check(PrepareInput({4}, at::kFloat, buffers, InputAccess::kBorrowReadOnly,
                     device, &prepared)
            .ok());
  Check(prepared.data_ptr() == source.data_ptr());
  Check(PrepareInput({4}, at::kFloat, buffers,
                     InputAccess::kRequireOwnedStorage, device, &prepared)
            .ok());
  Check(prepared.data_ptr() != source.data_ptr());
  prepared.add_(10);
  Check(at::equal(source.cpu(), original));
  const std::vector<InputBuffer> fragments = {
      {source.data_ptr(), 2 * sizeof(float), device},
      {source.data_ptr<float>() + 2, 2 * sizeof(float), device}};
  Check(PrepareInput({4}, at::kFloat, fragments, InputAccess::kBorrowReadOnly,
                     device, &prepared)
            .ok());
  Check(prepared.data_ptr() != source.data_ptr());
  Check(at::equal(prepared.cpu(), original));
  const std::vector<InputBuffer> incomplete = {
      {source.data_ptr(), sizeof(float), device}};
  Check(!PrepareInput({4}, at::kFloat, incomplete, InputAccess::kBorrowReadOnly,
                      device, &prepared)
             .ok());
  if (device.is_cuda()) {
    const std::vector<InputBuffer> host = {
        {original.data_ptr(), original.nbytes(), c10::Device(c10::kCPU)}};
    Check(PrepareInput({4}, at::kFloat, host, InputAccess::kRequireOwnedStorage,
                       device, &prepared)
              .ok());
    Check(prepared.is_cuda());
    prepared.add_(10);
    Check(at::equal(original, at::arange(4, original.options())));
    Check(at::equal(prepared.cpu(), original.add(10)));
  }
}

}  // namespace

int main(int argc, char**) {
  Analysis();
  Inputs(c10::Device(c10::kCPU));
  if (argc > 1) {
    const auto stream = c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard guard(stream);
    Inputs(c10::Device(c10::kCUDA, 0));
    stream.synchronize();
  }
  std::cout << "Backend analysis and tensor ownership checks passed\n";
}
