// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_RUNTIME_H_
#define LEAPP_RUNTIME_H_

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "leapp/bundle.h"
#include "leapp/status.h"
#include "leapp/dlpack.h"

namespace lrt {

enum class ExecutionDevice { kCpu, kCuda };

enum class MemoryType : uint8_t { kCpu, kPinnedCpu, kCuda };
struct MemoryLocation {
  MemoryType type = MemoryType::kCpu;
  uint32_t device_id = 0;
  bool operator==(const MemoryLocation&) const = default;
};

struct OutputPoolOptions {
  // Zero disables caching. The byte budget follows bundle and dynamic sizes.
  std::size_t max_entries = 64;
};

struct SessionOptions {
  ExecutionDevice device = ExecutionDevice::kCpu;
  uint32_t device_id = 0;
  MemoryLocation output_location;
  std::unordered_map<std::string, MemoryLocation> output_overrides;
  OutputPoolOptions output_pool;
};

/// @brief Retained outputs; copies share allocations.
class RunResult {
 public:
  RunResult() = default;

  bool is_valid() const { return data_ != nullptr; }

  /// @brief Borrowed names survive while any copy retains this result.
  std::vector<std::string_view> names() const;

  /// @brief Return a fresh managed DLPack lease over shared output storage. Failure preserves
  /// *tensor.
  Status GetTensor(std::string_view name, ManagedTensor* tensor) const;

 private:
  friend class Session;
  struct Data;
  std::shared_ptr<const Data> data_;
};

/// @brief Independent feedback state. Run/Reset serialize through the shared
/// Rust layer. Moving or destroying a handle requires no concurrent operations
/// on that handle.
class Session {
 public:
  Session();

  ~Session();

  Session(Session&&) noexcept;

  Session& operator=(Session&&) noexcept;

  Session(const Session&) = delete;

  Session& operator=(const Session&) = delete;

  /// @brief Load models and capture initial state into an empty destination.
  static Status Open(const Bundle& bundle, const SessionOptions& options,
                     Session* session);
  /// @brief Create fresh state over the same engine and immutable configuration.
  Status NewSession(Session* session) const;
  /// @brief Wait for this session's active call and close it. Peers remain usable.
  Status Close();
  /// @brief Retained snapshot, also available after Close.
  const Bundle& bundle() const { return bundle_; }

  /// @brief Borrow every public input for synchronous inference.
  /// Failure preserves *result and explicit feedback state. Backend-internal
  /// state changes cannot be rolled back. Success replaces *result.
  Status Run(std::span<const NamedTensor> inputs, RunResult* result);

  Status Run(std::initializer_list<NamedTensor> inputs,
             RunResult* result) {
    return Run(std::span<const NamedTensor>(inputs.begin(), inputs.size()),
               result);
  }

  /// @brief Restore explicit feedback to its initial state.
  Status Reset();

 private:
  Bundle bundle_;
  struct Data;
  std::unique_ptr<Data> data_;
};

}  // namespace lrt

#endif  // LEAPP_RUNTIME_H_
