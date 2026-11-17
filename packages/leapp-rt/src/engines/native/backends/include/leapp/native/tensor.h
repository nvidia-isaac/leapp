// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_NATIVE_TENSOR_H_
#define LEAPP_NATIVE_TENSOR_H_

#include <dlpack/dlpack.h>

#include <exception>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "leapp/data_type.h"
#include "leapp/native/detail/abi.h"
#include "leapp/status.h"

namespace lrt::native {

/// Owned port metadata. Shape dimensions of -1 are unresolved declarations.
struct PortSpec {
  std::string name;
  DataType dtype;
  std::vector<int64_t> shape;
};

/// Convert scalar types explicitly; public enum ordinals are not DLPack codes.
DLDataType dl_type(DataType dtype);
DataType data_type(DLDataType dtype);

/// Read-only, ready, contiguous storage and metadata borrowed for an operation.
/// Copying this wrapper copies the descriptor, not its shape or storage. Inputs
/// may alias. Do not mutate them, retain invocation pointers, or leave SDK work
/// accessing them after return or unwinding. CUDA addresses are not host
/// pointers.
class BorrowedTensor {
 public:
  explicit BorrowedTensor(const DLTensor& descriptor)
      : descriptor_(descriptor) {}

  const DLTensor& descriptor() const noexcept { return descriptor_; }

  const void* data() const noexcept {
    return reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(descriptor_.data) +
        descriptor_.byte_offset);
  }

  std::span<const int64_t> shape() const noexcept {
    return {descriptor_.shape, static_cast<size_t>(descriptor_.ndim)};
  }

  DataType dtype() const { return data_type(descriptor_.dtype); }

  size_t bytes() const;

 private:
  DLTensor descriptor_;
};

/// Writable contiguous destination, disjoint from inputs and other outputs.
/// Metadata is borrowed. Prepared storage lives until plugin destruction;
/// invocation output bindings live only through their synchronous call.
class WritableTensor {
 public:
  explicit WritableTensor(const DLTensor& descriptor)
      : descriptor_(descriptor) {}

  const DLTensor& descriptor() const noexcept { return descriptor_; }

  void* data() const noexcept {
    return const_cast<void*>(BorrowedTensor(descriptor_).data());
  }

  std::span<const int64_t> shape() const noexcept {
    return BorrowedTensor(descriptor_).shape();
  }

  DataType dtype() const { return data_type(descriptor_.dtype); }

  size_t bytes() const { return BorrowedTensor(descriptor_).bytes(); }

 private:
  DLTensor descriptor_;
};

/// Boundary input in the Session's input order, matching Rust's named
/// arguments.
struct NamedTensor {
  std::string_view name;
  BorrowedTensor tensor;
};

/// Replace prepared argument position with Session boundary input index.
using InputRoute = detail::InputRoute;

/// Override one declared output position for this invocation only.
struct OutputBinding {
  size_t position;
  WritableTensor tensor;
};

/// A failure with an explicit category, for SDK code that throws exceptions.
/// The adapter converts it to a Status; other exceptions become Internal
/// errors.
class BackendError : public std::exception {
 public:
  explicit BackendError(Status status) : status_(std::move(status)) {}

  const char* what() const noexcept override {
    return status_.message().c_str();
  }

  const Status& status() const noexcept { return status_; }

 private:
  Status status_;
};

}  // namespace lrt::native

#endif  // LEAPP_NATIVE_TENSOR_H_
