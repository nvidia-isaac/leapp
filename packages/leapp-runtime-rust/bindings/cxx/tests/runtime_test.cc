// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "leapp/runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>

namespace {
void Check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::cerr << "line " << line << ": " << expression << "\n";
    std::abort();
  }
}

#define CHECK(value) Check((value), #value, __LINE__)

void Ok(leapp::Status status) {
  if (!status.ok()) {
    std::cerr << status.phase() << ": " << status.message() << "\n";
    std::abort();
  }
}

leapp::Runtime Open(const std::filesystem::path& fixtures,
                    const std::filesystem::path& backends,
                    std::string_view name, bool cuda = false) {
  leapp::Bundle bundle;
  Ok(leapp::Bundle::Open(fixtures / (std::string(name) + ".yaml"), &bundle));
  leapp::Runtime runtime;
  leapp::RuntimeOptions options;
  options.backend_directory = backends;
  options.device =
      cuda ? leapp::ExecutionDevice::kCuda : leapp::ExecutionDevice::kCpu;
  Ok(leapp::Runtime::Open(bundle, options, &runtime));
  return runtime;
}

leapp::Tensor Step(leapp::Session& session, float value = 1,
                   leapp::MemoryLocation location = {}) {
  const std::array<float, 4> input = {value, value, value, value};
  leapp::RunResult result;
  leapp::RunOptions options;
  options.output_location = location;
  Ok(session.Run(
      {{"accumulator/observation", leapp::TensorView::cpu<float>(input, {4})}},
      options, &result));
  CHECK(result.names() == std::vector<std::string_view>{"accumulator/next"});
  leapp::Tensor tensor;
  Ok(result.GetTensor("accumulator/next", &tensor));
  return tensor;
}

std::vector<float> Values(const leapp::Tensor& tensor) {
  std::vector<std::byte> bytes;
  Ok(tensor.CopyToCpu(&bytes));
  std::vector<float> result(bytes.size() / sizeof(float));
  if (!bytes.empty()) {
    std::memcpy(result.data(), bytes.data(), bytes.size());
  }
  return result;
}

void Expect(const leapp::Tensor& tensor, float value) {
  CHECK(Values(tensor) == std::vector<float>(4, value));
}

void FeedbackAndLifetimes(const std::filesystem::path& fixtures,
                          const std::filesystem::path& backends) {
  auto runtime = Open(fixtures, backends, "feedback");
  leapp::Session session;
  Ok(runtime.CreateSession(&session));
  CHECK(!runtime.CreateSession(&session).ok());
  auto first = Step(session);
  auto shared = first;
  std::span<const float> borrowed;
  Ok(first.CpuSpan(&borrowed));
  CHECK(borrowed[0] == 1);
  Expect(Step(session), 2);
  Expect(first, 1);
  Ok(session.Reset());
  Expect(Step(session), 1);
  Ok(runtime.Close());
  Ok(runtime.Close());
  CHECK(runtime.bundle().model_names().front() == "accumulator");
  CHECK(!session.Reset().ok());
  leapp::Session closed;
  CHECK(!runtime.CreateSession(&closed).ok());
  first = {};
  runtime = {};
  session = {};
  Expect(shared, 1);
  CHECK(borrowed[0] == 1);

  // Destruction alone leaves a retained session usable.
  auto other = Open(fixtures, backends, "feedback");
  Ok(other.CreateSession(&session));
  other = {};
  Expect(Step(session), 1);
}

void InvalidInputs(const std::filesystem::path& fixtures,
                   const std::filesystem::path& backends) {
  auto runtime = Open(fixtures, backends, "dynamic");
  leapp::Session session;
  Ok(runtime.CreateSession(&session));
  const std::array<float, 4> data = {1, 2, 3, 4};
  leapp::RunResult result;
  auto view = leapp::TensorView::cpu<float>(data, {4});
  Ok(session.Run({{"dynamic/input", view}}, {}, &result));
  leapp::Tensor retained;
  Ok(result.GetTensor("dynamic/output", &retained));
  const auto original = retained.data();
  std::span<const double> mismatched;
  CHECK(!retained.CpuSpan(&mismatched).ok());

  for (int scenario = 0; scenario < 7; ++scenario) {
    auto invalid = view;
    std::string name = "dynamic/input";
    switch (scenario) {
      case 0:
        invalid.byte_size -= 1;
        break;
      case 1:
        invalid.shape = {-1};
        break;
      case 2:
        invalid.data = nullptr;
        break;
      case 3:
        invalid.dtype = static_cast<leapp::DataType>(255);
        break;
      case 4:
        invalid.location.device_id = 1;
        break;
      case 5:
        name = "missing/input";
        break;
      case 6:
        name = std::string(1, '\xff');
        break;
    }
    auto status = session.Run({{name, invalid}}, {}, &result);
    CHECK(!status.ok());
    leapp::Tensor preserved;
    Ok(result.GetTensor("dynamic/output", &preserved));
    CHECK(preserved.data() == original);
  }
  CHECK(
      !session
           .Run({{"dynamic/input", view}, {"dynamic/input", view}}, {}, &result)
           .ok());
  CHECK(!session.Run({}, {}, &result).ok());
  CHECK(!result.GetTensor("missing", &retained).ok());
  CHECK(retained.data() == original);
  leapp::RunOptions options;
  options.output_overrides.emplace("missing", leapp::MemoryLocation{});
  CHECK(!session.Run({{"dynamic/input", view}}, options, &result).ok());
  Ok(session.Run({{"dynamic/input", retained.view()}}, {}, &result));
  leapp::Tensor reused;
  Ok(result.GetTensor("dynamic/output", &reused));
  CHECK(Values(reused) == std::vector<float>(data.begin(), data.end()));
  const std::span<const float> empty;
  Ok(session.Run({{"dynamic/input", leapp::TensorView::cpu<float>(empty, {0})}},
                 {}, &result));
  Ok(result.GetTensor("dynamic/output", &reused));
  CHECK(reused.byte_size() == 0);
  CHECK(Values(reused).empty());
}

void ConcurrentSessions(const std::filesystem::path& fixtures,
                        const std::filesystem::path& backends) {
  auto runtime = Open(fixtures, backends, "feedback");
  leapp::Session session;
  Ok(runtime.CreateSession(&session));
  std::vector<float> values(12);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < values.size(); ++i) {
    threads.emplace_back([&, i] { values[i] = Values(Step(session))[0]; });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  std::sort(values.begin(), values.end());
  for (size_t i = 0; i < values.size(); ++i) {
    CHECK(values[i] == i + 1);
  }
  threads.clear();
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&] {
      leapp::Session independent;
      Ok(runtime.CreateSession(&independent));
      Expect(Step(independent), 1);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  std::atomic<bool> started = false;
  std::thread worker([&] {
    const std::array<float, 4> data = {1, 1, 1, 1};
    started = true;
    for (int i = 0; i < 50; ++i) {
      leapp::RunResult result;
      auto status = session.Run({{"accumulator/observation",
                                  leapp::TensorView::cpu<float>(data, {4})}},
                                {}, &result);
      if (!status.ok()) {
        CHECK(status.code() == leapp::ErrorCode::kInvalidArgument);
        return;
      }
    }
  });
  while (!started) {
    std::this_thread::yield();
  }
  Ok(runtime.Close());
  worker.join();
}

void CudaBuffers(const std::filesystem::path& fixtures,
                 const std::filesystem::path& backends) {
  auto runtime = Open(fixtures, backends, "feedback", true);
  leapp::Session session;
  Ok(runtime.CreateSession(&session));
  const std::array<float, 4> data = {1, 2, 3, 4};
  void* device = nullptr;
  void* pinned = nullptr;
  CHECK(cudaMalloc(&device, sizeof(data)) == cudaSuccess);
  CHECK(cudaMallocHost(&pinned, sizeof(data)) == cudaSuccess);
  std::memcpy(pinned, data.data(), sizeof(data));
  CHECK(cudaMemcpy(device, data.data(), sizeof(data), cudaMemcpyHostToDevice) ==
        cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  leapp::Tensor retained;
  for (auto input_type :
       {leapp::MemoryType::kCpu, leapp::MemoryType::kPinnedCpu,
        leapp::MemoryType::kCuda}) {
    const void* input = input_type == leapp::MemoryType::kCpu ? data.data()
                        : input_type == leapp::MemoryType::kPinnedCpu ? pinned
                                                                      : device;
    for (auto output_type :
         {leapp::MemoryType::kCpu, leapp::MemoryType::kPinnedCpu,
          leapp::MemoryType::kCuda}) {
      Ok(session.Reset());
      leapp::TensorView view{
          leapp::DataType::kFloat32, {4}, input, sizeof(data), {input_type, 0}};
      leapp::RunOptions options;
      options.output_overrides.emplace("accumulator/next",
                                       leapp::MemoryLocation{output_type, 0});
      leapp::RunResult result;
      Ok(session.Run({{"accumulator/observation", view}}, options, &result));
      Ok(result.GetTensor("accumulator/next", &retained));
      CHECK(retained.location().type == output_type);
      CHECK(Values(retained) == std::vector<float>(data.begin(), data.end()));
      std::span<const float> span;
      CHECK(retained.CpuSpan(&span).ok() ==
            (output_type != leapp::MemoryType::kCuda));
    }
  }
  CHECK(cudaFree(device) == cudaSuccess);
  CHECK(cudaFreeHost(pinned) == cudaSuccess);
  Ok(runtime.Close());
  CHECK(Values(retained) == std::vector<float>(data.begin(), data.end()));
}

}  // namespace

int main(int argc, char** argv) {
  static_assert(!std::is_copy_constructible_v<leapp::Runtime>);
  static_assert(!std::is_copy_constructible_v<leapp::Session>);
  static_assert(std::is_copy_constructible_v<leapp::Tensor>);
  CHECK(argc == 4);
  const std::filesystem::path fixtures = argv[1];
  const std::filesystem::path backends = argv[2];
  FeedbackAndLifetimes(fixtures, backends);
  InvalidInputs(fixtures, backends);
  ConcurrentSessions(fixtures, backends);
  if (std::string_view(argv[3]) == "ON") {
    CudaBuffers(fixtures, backends);
  }
  std::cout << "C++ binding integration checks passed\n";
}
