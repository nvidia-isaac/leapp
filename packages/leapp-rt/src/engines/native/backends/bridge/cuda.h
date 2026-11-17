// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace leapp_native::cuda {

inline void check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("native CUDA: ") +
                             cudaGetErrorString(status));
  }
}

// Destruction also selects the owning device, without throwing on device loss.
class DeviceScope {
 public:
  explicit DeviceScope(int device) noexcept {
    status_ = cudaGetDevice(&previous_);
    if (status_ == cudaSuccess && previous_ != device) {
      status_ = cudaSetDevice(device);
    }
  }

  ~DeviceScope() {
    if (previous_ >= 0) {
      cudaSetDevice(previous_);
    }
  }

  void check() const { cuda::check(status_); }

 private:
  int previous_ = -1;
  cudaError_t status_;
};

struct Stream {
  explicit Stream(int id, cudaStream_t external = nullptr)
      : device(id), value(external), owned(external == nullptr) {
    DeviceScope scope(device);
    scope.check();
    if (owned) {
      check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }
  }

  ~Stream() {
    if (!owned) {
      return;
    }
    DeviceScope scope(device);
    cudaStreamSynchronize(value);
    cudaStreamDestroy(value);
  }

  int device;
  cudaStream_t value = nullptr;
  bool owned;
};

// Declare after owners of borrowed buffers so exceptions drain work first.
class Work {
 public:
  explicit Work(Stream& stream) : device_(stream.device), stream_(stream) {
    device_.check();
  }

  ~Work() {
    if (!complete_) {
      // Failed native calls may leave work on auxiliary/worker streams.
      // Drain all of it before releasing borrowed buffers.
      cudaDeviceSynchronize();
    }
  }

  void finish(bool background_work = false) {
    if (background_work) {
      check(cudaDeviceSynchronize());
    } else {
      check(cudaStreamSynchronize(stream_.value));
    }
    complete_ = true;
  }

 private:
  DeviceScope device_;
  Stream& stream_;
  bool complete_ = false;
};

}  // namespace leapp_native::cuda
