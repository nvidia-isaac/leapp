// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_NATIVE_TRANSFERS_H_
#define LEAPP_NATIVE_TRANSFERS_H_

#include <optional>
#include <utility>
#include <vector>

#include "leapp/native/tensor.h"

namespace lrt::native {

namespace detail {

struct Access;

}  // namespace detail

/// Independent runtime storage. Moving transfers ownership; destruction is safe
/// on any host thread. Results survive later runs and plugin/library
/// destruction. Only Transfers can create a populated owner; foreign SDK
/// allocations remain borrowed until copied. An empty owner represents no
/// tensor.
class OwnedTensor {
 public:
  OwnedTensor() = default;
  OwnedTensor(const OwnedTensor&) = delete;
  OwnedTensor& operator=(const OwnedTensor&) = delete;

  OwnedTensor(OwnedTensor&& other) noexcept
      : handle_(std::exchange(other.handle_, {})) {}

  OwnedTensor& operator=(OwnedTensor&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = std::exchange(other.handle_, {});
    }
    return *this;
  }

  ~OwnedTensor() { reset(); }

  explicit operator bool() const noexcept { return handle_.value != nullptr; }

 private:
  friend class Transfers;
  friend struct detail::Access;

  void reset() noexcept {
    if (handle_.value) {
      handle_.destroy(handle_.value);
    }
    handle_ = {};
  }

  detail::TensorHandle handle_{};
};

/// One entry per declared output. nullopt means the supplied destination was
/// written; every other entry must contain a populated independent owner.
using DynamicOutputs = std::vector<std::optional<OwnedTensor>>;

/// Workspace transfers borrowed exclusively for run_dynamic. Never retain this
/// object, its callbacks, or context, or call it from a background thread.
class Transfers {
 public:
  /// Snapshot ready contiguous bytes to the workspace execution device. Source
  /// metadata/storage need only survive this synchronous operation. Complete
  /// SDK producer work, including auxiliary streams, before calling. The
  /// destination must be empty; failure leaves it empty. No backend allocator
  /// is required.
  // NOLINTNEXTLINE(build/include_what_you_use)
  Status copy(BorrowedTensor source, OwnedTensor* destination);

 private:
  friend struct detail::Access;

  explicit Transfers(detail::Host host) : host_(host) {}

  detail::Host host_;
};

}  // namespace lrt::native

#endif  // LEAPP_NATIVE_TRANSFERS_H_
