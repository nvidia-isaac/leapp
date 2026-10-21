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

leapp::Session Open(const std::filesystem::path& fixtures,
                    std::string_view name, bool cuda = false,
                    leapp::MemoryLocation output_location = {}) {
  leapp::Bundle bundle;
  Ok(leapp::Bundle::Open(fixtures / (std::string(name) + ".yaml"), &bundle));
  leapp::Session runtime;
  leapp::SessionOptions options;
  options.output_overrides.emplace("accumulator/next", output_location);
  if (name != "feedback") options.output_overrides.clear();
  options.device =
      cuda ? leapp::ExecutionDevice::kCuda : leapp::ExecutionDevice::kCpu;
  Ok(leapp::Session::Open(bundle, options, &runtime));
  return runtime;
}

DLTensor Cpu(std::span<const float> data, int64_t* shape) {
  return {const_cast<float*>(data.data()), {kDLCPU, 0}, 1,
          {kDLFloat, 32, 1}, shape, nullptr, 0};
}

leapp::ManagedTensor Step(leapp::Session& session, float value = 1) {
  const std::array<float, 4> input = {value, value, value, value};
  int64_t shape = 4;
  auto descriptor = Cpu(input, &shape);
  leapp::RunResult result;
  Ok(session.Run(
      {{"accumulator/observation", &descriptor}},
      &result));
  CHECK(result.names() == std::vector<std::string_view>{"accumulator/next"});
  leapp::ManagedTensor tensor;
  Ok(result.GetTensor("accumulator/next", &tensor));
  return tensor;
}

std::vector<float> Values(const leapp::ManagedTensor& tensor) {
  const auto& t = tensor->dl_tensor;
  CHECK(t.dtype.code == kDLFloat && t.dtype.bits == 32 && t.dtype.lanes == 1);
  size_t size = 1;
  for (int i = 0; i < t.ndim; ++i) size *= t.shape[i];
  std::vector<float> result(size);
  if (size != 0) {
    auto* data = static_cast<const std::byte*>(t.data) + t.byte_offset;
    if (t.device.device_type == kDLCUDA) {
      CHECK(cudaMemcpy(result.data(), data, size * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    } else {
      std::memcpy(result.data(), data, size * sizeof(float));
    }
  }
  return result;
}

void Expect(const leapp::ManagedTensor& tensor, float value) {
  CHECK(Values(tensor) == std::vector<float>(4, value));
}

void FeedbackAndLifetimes(const std::filesystem::path& fixtures) {
  auto runtime = Open(fixtures, "feedback");
  leapp::Session session;
  Ok(runtime.NewSession(&session));
  CHECK(!runtime.NewSession(&session).ok());
  auto first = Step(session);
  std::span<const float> borrowed(static_cast<const float*>(first->dl_tensor.data), 4);
  CHECK(borrowed[0] == 1);
  static_cast<float*>(first->dl_tensor.data)[0] = 99;
  Expect(Step(session), 2);
  CHECK(Values(first)[0] == 99);
  Ok(session.Reset());
  Expect(Step(session), 1);
  Ok(runtime.Close());
  Ok(runtime.Close());
  CHECK(runtime.bundle().model_names().front() == "accumulator");
  Expect(Step(session), 2);
  leapp::Session fresh;
  Ok(session.NewSession(&fresh));
  Expect(Step(fresh), 1);
  Ok(session.Close());
  CHECK(!session.Reset().ok());
  Expect(Step(fresh), 2);
  leapp::Session closed;
  CHECK(!runtime.NewSession(&closed).ok());
  auto shared = std::move(first);
  runtime = {};
  session = {};
  CHECK(Values(shared)[0] == 99);
  CHECK(borrowed[0] == 99);

  // Destruction alone leaves a retained session usable.
  auto other = Open(fixtures, "feedback");
  Ok(other.NewSession(&session));
  other = {};
  Expect(Step(session), 1);
}

void InvalidInputs(const std::filesystem::path& fixtures) {
  auto runtime = Open(fixtures, "dynamic");
  leapp::Session session;
  Ok(runtime.NewSession(&session));
  const std::array<float, 4> data = {1, 2, 3, 4};
  leapp::RunResult result;
  int64_t shape = 4;
  auto view = Cpu(data, &shape);
  Ok(session.Run({{"dynamic/input", &view}}, &result));
  leapp::ManagedTensor retained;
  Ok(result.GetTensor("dynamic/output", &retained));
  const auto original = retained->dl_tensor.data;


  for (int scenario = 0; scenario < 7; ++scenario) {
    auto invalid = view;
    int64_t bad_shape = -1;
    int64_t bad_stride = 2;
    std::string name = "dynamic/input";
    switch (scenario) {
      case 0:
        invalid.strides = &bad_stride;
        break;
      case 1:
        invalid.shape = &bad_shape;
        break;
      case 2:
        invalid.data = nullptr;
        break;
      case 3:
        invalid.dtype.code = 255;
        break;
      case 4:
        invalid.device.device_id = 1;
        break;
      case 5:
        name = "missing/input";
        break;
      case 6:
        name = std::string(1, '\xff');
        break;
    }
    auto status = session.Run({{name, &invalid}}, &result);
    CHECK(!status.ok());
    leapp::ManagedTensor preserved;
    Ok(result.GetTensor("dynamic/output", &preserved));
    CHECK(preserved->dl_tensor.data == original);
  }
  CHECK(
      !session
           .Run({{"dynamic/input", &view}, {"dynamic/input", &view}}, &result)
           .ok());
  CHECK(!session.Run({}, &result).ok());
  CHECK(!result.GetTensor("missing", &retained).ok());
  CHECK(retained->dl_tensor.data == original);
  leapp::SessionOptions options;
  options.output_overrides.emplace("missing", leapp::MemoryLocation{});
  leapp::Session invalid_session;
  CHECK(!leapp::Session::Open(session.bundle(), options, &invalid_session).ok());
  Ok(session.Run({{"dynamic/input", &retained->dl_tensor}}, &result));
  leapp::ManagedTensor reused;
  Ok(result.GetTensor("dynamic/output", &reused));
  CHECK(Values(reused) == std::vector<float>(data.begin(), data.end()));
  const std::span<const float> empty;
  int64_t empty_shape = 0;
  auto empty_descriptor = Cpu(empty, &empty_shape);
  Ok(session.Run({{"dynamic/input", &empty_descriptor}},
                 &result));
  Ok(result.GetTensor("dynamic/output", &reused));
  CHECK(reused->dl_tensor.shape[0] == 0);
  CHECK(Values(reused).empty());
}

void ConcurrentSessions(const std::filesystem::path& fixtures) {
  auto runtime = Open(fixtures, "feedback");
  leapp::Session session;
  Ok(runtime.NewSession(&session));
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
      Ok(runtime.NewSession(&independent));
      Expect(Step(independent), 1);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  std::atomic<bool> started = false;
  std::thread worker([&] {
    const std::array<float, 4> data = {1, 1, 1, 1};
    int64_t shape = 4;
    auto descriptor = Cpu(data, &shape);
    started = true;
    for (int i = 0; i < 50; ++i) {
      leapp::RunResult result;
      auto status = session.Run({{"accumulator/observation",
                                  &descriptor}},
                                &result);
      if (!status.ok()) {
        CHECK(status.code() == leapp::ErrorCode::kInvalidArgument);
        return;
      }
    }
  });
  while (!started) {
    std::this_thread::yield();
  }
  Ok(session.Close());
  worker.join();
  Expect(Step(runtime), 1);
}

void CudaBuffers(const std::filesystem::path& fixtures) {
  const std::array<float, 4> data = {1, 2, 3, 4};
  void* device = nullptr;
  void* pinned = nullptr;
  CHECK(cudaMalloc(&device, sizeof(data)) == cudaSuccess);
  CHECK(cudaMallocHost(&pinned, sizeof(data)) == cudaSuccess);
  std::memcpy(pinned, data.data(), sizeof(data));
  CHECK(cudaMemcpy(device, data.data(), sizeof(data), cudaMemcpyHostToDevice) ==
        cudaSuccess);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  leapp::ManagedTensor retained;
  for (auto output_type :
       {leapp::MemoryType::kCpu, leapp::MemoryType::kPinnedCpu,
        leapp::MemoryType::kCuda}) {
    auto session = Open(fixtures, "feedback", true, {output_type, 0});
    for (auto input_type :
         {leapp::MemoryType::kCpu, leapp::MemoryType::kPinnedCpu,
          leapp::MemoryType::kCuda}) {
      const void* input = input_type == leapp::MemoryType::kCpu ? data.data()
                          : input_type == leapp::MemoryType::kPinnedCpu ? pinned
                                                                        : device;
      Ok(session.Reset());
      int64_t shape = 4;
      const auto device_type = input_type == leapp::MemoryType::kCpu ? kDLCPU :
          input_type == leapp::MemoryType::kPinnedCpu ? kDLCUDAHost : kDLCUDA;
      DLTensor view{const_cast<void*>(input), {device_type, 0}, 1,
                    {kDLFloat, 32, 1}, &shape, nullptr, 0};
      leapp::RunResult result;
      Ok(session.Run({{"accumulator/observation", &view}}, &result));
      Ok(result.GetTensor("accumulator/next", &retained));
      const auto expected_device = output_type == leapp::MemoryType::kCpu ? kDLCPU :
          output_type == leapp::MemoryType::kPinnedCpu ? kDLCUDAHost : kDLCUDA;
      CHECK(retained->dl_tensor.device.device_type == expected_device);
      CHECK(Values(retained) == std::vector<float>(data.begin(), data.end()));

    }
  }
  CHECK(cudaFree(device) == cudaSuccess);
  CHECK(cudaFreeHost(pinned) == cudaSuccess);
  CHECK(Values(retained) == std::vector<float>(data.begin(), data.end()));
}

}  // namespace

int main(int argc, char** argv) {
  static_assert(!std::is_copy_constructible_v<leapp::Session>);
  static_assert(!std::is_copy_constructible_v<leapp::ManagedTensor>);
  CHECK(argc == 3);
  const std::filesystem::path fixtures = argv[1];
  FeedbackAndLifetimes(fixtures);
  InvalidInputs(fixtures);
  ConcurrentSessions(fixtures);
  if (std::string_view(argv[2]) == "ON") {
    CudaBuffers(fixtures);
  }
  std::cout << "C++ binding integration checks passed\n";
}
