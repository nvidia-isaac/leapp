// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_RUNTIME_H_
#define LEAPP_RUNTIME_H_

#include <filesystem>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "leapp/bundle.h"
#include "leapp/status.h"
#include "leapp/tensor.h"

namespace leapp {

enum class ExecutionDevice { kCpu, kCuda };

struct RuntimeOptions {
  ExecutionDevice device = ExecutionDevice::kCpu;
  uint32_t device_id = 0;
  std::filesystem::path backend_directory = "/opt/tritonserver/backends";
};

struct RunOptions {
  MemoryLocation output_location;
  std::unordered_map<std::string, MemoryLocation> output_overrides;
};

/// @brief Retained immutable outputs; copies share allocations.
class RunResult {
 public:
  RunResult() = default;

  bool is_valid() const { return data_ != nullptr; }

  /// @brief Borrowed names survive while any copy retains this result.
  std::vector<std::string_view> names() const;

  /// @brief Return an independent owner of shared storage. Failure preserves
  /// *tensor.
  Status GetTensor(std::string_view name, Tensor* tensor) const;

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

  /// @brief Borrow every public input for synchronous inference.
  /// Failure preserves *result and explicit feedback state. Backend-internal
  /// state changes cannot be rolled back. Success replaces *result.
  Status Run(std::span<const NamedTensor> inputs, const RunOptions& options,
             RunResult* result);

  Status Run(std::initializer_list<NamedTensor> inputs,
             const RunOptions& options, RunResult* result) {
    return Run(std::span<const NamedTensor>(inputs.begin(), inputs.size()),
               options, result);
  }

  /// @brief Restore explicit feedback to its initial state.
  Status Reset();

 private:
  friend class Runtime;
  struct Data;
  std::unique_ptr<Data> data_;
};

/// @brief Loaded models retained by sessions. Explicit Close invalidates
/// sessions. Close can overlap Run/Reset and waits for active native calls.
/// Destruction or moving requires external synchronization; destruction alone
/// does not close sessions.
class Runtime {
 public:
  Runtime();

  ~Runtime();

  Runtime(Runtime&&) noexcept;

  Runtime& operator=(Runtime&&) noexcept;

  Runtime(const Runtime&) = delete;

  Runtime& operator=(const Runtime&) = delete;

  /// @brief Load a bundle into an empty destination; failure preserves
  /// *runtime.
  static Status Open(const Bundle& bundle, const RuntimeOptions& options,
                     Runtime* runtime);

  /// @brief Retained snapshot, also available after Close. Empty before Open.
  const Bundle& bundle() const { return bundle_; }

  /// @brief Initialize an empty session destination; failure preserves it.
  Status CreateSession(Session* session) const;

  /// @brief Wait for active calls and disable sessions. Repeated calls succeed.
  /// Poisoned locks are reported after cleanup completes.
  Status Close();

 private:
  Bundle bundle_;
  struct Data;
  std::unique_ptr<Data> data_;
};

}  // namespace leapp

#endif  // LEAPP_RUNTIME_H_
