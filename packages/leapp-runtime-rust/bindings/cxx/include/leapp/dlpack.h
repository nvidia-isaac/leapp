// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef LEAPP_DLPACK_H_
#define LEAPP_DLPACK_H_
#include <dlpack/dlpack.h>
#include <memory>
#include <string_view>

namespace leapp {
/// Borrowed descriptor, metadata, and storage must stay valid and unmodified,
/// with producer work complete on every execution stream, until Run returns.
struct NamedTensor {
  std::string_view name;
  const DLTensor* tensor;
};
struct DLPackDeleter {
  void operator()(DLManagedTensorVersioned* tensor) const noexcept {
    if (tensor != nullptr && tensor->deleter != nullptr) {
      tensor->deleter(tensor);
    }
  }
};
/// Standard DLPack lease with RAII cleanup. release() transfers it to a consumer.
using ManagedTensor = std::unique_ptr<DLManagedTensorVersioned, DLPackDeleter>;
}  // namespace leapp
#endif  // LEAPP_DLPACK_H_
