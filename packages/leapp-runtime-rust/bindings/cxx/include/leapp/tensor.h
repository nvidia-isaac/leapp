// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_TENSOR_H_
#define LEAPP_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "leapp/data_type.h"
#include "leapp/status.h"

namespace leapp {

enum class MemoryType : uint8_t { kCpu, kPinnedCpu, kCuda };

/// @brief Buffer placement, independent of model execution placement.
struct MemoryLocation {
  MemoryType type = MemoryType::kCpu;
  uint32_t device_id = 0;  ///< Host locations require zero.

  bool operator==(const MemoryLocation&) const = default;
};

namespace detail {

template <typename T>
struct ElementType;

template <>
struct ElementType<uint8_t> {
  static constexpr DataType value = DataType::kUint8;
};

template <>
struct ElementType<uint16_t> {
  static constexpr DataType value = DataType::kUint16;
};

template <>
struct ElementType<uint32_t> {
  static constexpr DataType value = DataType::kUint32;
};

template <>
struct ElementType<uint64_t> {
  static constexpr DataType value = DataType::kUint64;
};

template <>
struct ElementType<int8_t> {
  static constexpr DataType value = DataType::kInt8;
};

template <>
struct ElementType<int16_t> {
  static constexpr DataType value = DataType::kInt16;
};

template <>
struct ElementType<int32_t> {
  static constexpr DataType value = DataType::kInt32;
};

template <>
struct ElementType<int64_t> {
  static constexpr DataType value = DataType::kInt64;
};

template <>
struct ElementType<float> {
  static constexpr DataType value = DataType::kFloat32;
};

template <>
struct ElementType<double> {
  static constexpr DataType value = DataType::kFloat64;
};

}  // namespace detail

/// @brief Borrowed immutable data with an owned shape descriptor.
/// Data must remain alive, unmodified, and ready on all streams until Run
/// returns. The supplied byte size must exactly match the shape and dtype.
struct TensorView {
  DataType dtype = DataType::kFloat32;
  std::vector<int64_t> shape;
  const void* data = nullptr;
  size_t byte_size = 0;
  MemoryLocation location;

  /// @brief Borrow a contiguous CPU span. Run validates its shape and byte
  /// count.
  template <typename T>
  static TensorView cpu(std::span<const T> values, std::vector<int64_t> shape) {
    return {detail::ElementType<std::remove_cv_t<T>>::value,
            std::move(shape),
            values.data(),
            values.size_bytes(),
            {}};
  }
};

struct NamedTensor {
  std::string_view name;  ///< Qualified node/port name, borrowed through Run.
  TensorView tensor;
};

/// @brief Immutable output owner. Copies share storage beyond runtime closure.
class Tensor {
 public:
  Tensor() = default;

  bool is_valid() const { return data_ != nullptr; }

  DataType dtype() const;

  std::span<const int64_t> shape() const;

  MemoryLocation location() const;

  size_t byte_size() const;

  /// @brief Expose storage, permanently excluding it from output-pool reuse.
  /// Keep an owner alive; CUDA addresses must not be host-dereferenced.
  const void* data() const;

  /// @brief Borrow this tensor for input; keep its owner alive through Run.
  /// Exposes storage and permanently excludes it from output-pool reuse.
  TensorView view() const;

  /// @brief Borrow host values until this owner and all its copies are
  /// released. Failure, including CUDA storage or a dtype mismatch, preserves
  /// *values.
  template <typename T>
  Status CpuSpan(std::span<const T>* values) const {
    if (values == nullptr || !is_valid() ||
        location().type == MemoryType::kCuda ||
        dtype() != detail::ElementType<std::remove_cv_t<T>>::value ||
        reinterpret_cast<uintptr_t>(data()) % alignof(T) != 0) {
      return {ErrorCode::kInvalidArgument, "buffer",
              "expected a valid host tensor with matching scalar type"};
    }
    *values = {static_cast<const T*>(data()), byte_size() / sizeof(T)};
    return {};
  }

  /// @brief Copy packed bytes to CPU; CUDA transfers complete before return.
  /// Failure preserves *bytes.
  Status CopyToCpu(std::vector<std::byte>* bytes) const;

 private:
  friend class RunResult;
  struct Data;
  std::shared_ptr<const Data> data_;
};

}  // namespace leapp

#endif  // LEAPP_TENSOR_H_
