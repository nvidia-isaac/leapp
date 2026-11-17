// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// SDK-local compact views used by the Torch implementation.
// These are implementation details, not the plugin interface or cross-library
// ABI.
#include <array>
#include <vector>

#include "leapp/native/backend_plugin.h"

namespace leapp_native::tensors {

struct Port {
  const char* name;
  size_t name_size;
  uint8_t dtype;
  const int64_t* shape;
  size_t rank;
};

struct View {
  Port port;
  uintptr_t data;
  size_t bytes;
  int32_t device;
};

// SDK-owned dynamic results. Implementations retain storage and shape metadata
// until the plugin finishes copying these views into runtime-owned tensors.
struct Outputs {
  virtual ~Outputs() = default;

  std::vector<View> views;
};

using InputRoute = lrt::native::InputRoute;

struct OutputBinding {
  size_t position;
  View view;
};

inline constexpr std::array<lrt::DataType, 13> types = {
    lrt::DataType::kBool,     lrt::DataType::kUint8,   lrt::DataType::kUint16,
    lrt::DataType::kUint32,   lrt::DataType::kUint64,  lrt::DataType::kInt8,
    lrt::DataType::kInt16,    lrt::DataType::kInt32,   lrt::DataType::kInt64,
    lrt::DataType::kFloat16,  lrt::DataType::kFloat32, lrt::DataType::kFloat64,
    lrt::DataType::kBfloat16,
};

inline uint8_t dtype(lrt::DataType value) {
  for (size_t i = 0; i < types.size(); ++i) {
    if (types[i] == value) {
      return i;
    }
  }
  throw lrt::native::BackendError(
      {lrt::ErrorCode::kUnsupported, "dtype", "unsupported SDK dtype"});
}

inline std::vector<Port> ports(std::span<const lrt::native::PortSpec> values) {
  std::vector<Port> result;
  result.reserve(values.size());
  for (const auto& value : values) {
    result.push_back({value.name.data(), value.name.size(), dtype(value.dtype),
                      value.shape.data(), value.shape.size()});
  }
  return result;
}

inline View view(lrt::native::BorrowedTensor value) {
  const auto& tensor = value.descriptor();
  return {{nullptr, 0, dtype(value.dtype()), tensor.shape,
           static_cast<size_t>(tensor.ndim)},
          reinterpret_cast<uintptr_t>(value.data()),
          value.bytes(),
          tensor.device.device_type == kDLCPU ? -1 : tensor.device.device_id};
}

template <typename Tensor>
std::vector<View> views(std::span<const Tensor> values) {
  std::vector<View> result;
  result.reserve(values.size());
  for (const auto& value : values) {
    result.push_back(view(lrt::native::BorrowedTensor(value.descriptor())));
  }
  return result;
}

inline std::vector<OutputBinding> bindings(
    std::span<const lrt::native::OutputBinding> values) {
  std::vector<OutputBinding> result;
  result.reserve(values.size());
  for (const auto& value : values) {
    result.push_back({value.position, view(lrt::native::BorrowedTensor(
                                          value.tensor.descriptor()))});
  }
  return result;
}

inline DLTensor descriptor(const View& value) {
  return {reinterpret_cast<void*>(value.data),
          {value.device < 0 ? kDLCPU : kDLCUDA,
           value.device < 0 ? 0 : value.device},
          static_cast<int32_t>(value.port.rank),
          lrt::native::dl_type(types.at(value.port.dtype)),
          const_cast<int64_t*>(value.port.shape),
          nullptr,
          0};
}

// Own fixed metadata once. Invocation substitution retains no caller pointers.
class FixedViews {
 public:
  void assign(std::span<const lrt::native::BorrowedTensor> inputs) {
    values = views(inputs);
    shapes_.resize(values.size());
    addresses_.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      const auto shape = inputs[i].shape();
      shapes_[i].clear();
      if (!shape.empty()) {
        shapes_[i].assign(shape.begin(), shape.end());
      }
      values[i].port.shape = shapes_[i].data();
      addresses_[i] = values[i].data;
    }
  }

  struct Invocation {
    FixedViews& fixed;
    std::span<const InputRoute> routes;

    Invocation(FixedViews& owner,
               std::span<const lrt::native::NamedTensor> inputs,
               std::span<const InputRoute> selected)
        : fixed(owner), routes(selected) {
      for (const auto& route : routes) {
        fixed.values[route.position].data =
            reinterpret_cast<uintptr_t>(inputs[route.boundary].tensor.data());
      }
    }

    ~Invocation() {
      for (const auto& route : routes) {
        fixed.values[route.position].data = fixed.addresses_[route.position];
      }
    }
  };

  std::vector<View> values;

 private:
  std::vector<std::vector<int64_t>> shapes_;
  std::vector<uintptr_t> addresses_;
};

}  // namespace leapp_native::tensors
