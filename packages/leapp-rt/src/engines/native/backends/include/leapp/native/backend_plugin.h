// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file
/// @brief C++ backend plugin interface for the native runtime.

#ifndef LEAPP_NATIVE_BACKEND_PLUGIN_H_
#define LEAPP_NATIVE_BACKEND_PLUGIN_H_

#include <memory>

#include "leapp/native/transfers.h"

namespace lrt::native {

/// @brief Initialization metadata borrowed for the loader call.
///
/// Copy retained values. No CUDA SDK is required to use this type.
struct LoadContext {
  std::string_view model_path;  ///< Model artifact path.
  int32_t device;               ///< CPU (-1) or a CUDA device ordinal.

  /// @brief Caller-owned CUDA stream, or nullptr to request internal streams.
  /// Peers inherit this selection. A caller-owned stream must outlive all peers
  /// and SDK handle destruction.
  void* cuda_stream;

  std::span<const PortSpec> inputs;   ///< Declared inputs in model order.
  std::span<const PortSpec> outputs;  ///< Declared outputs in model order.
};

/// @brief One model node in one native workspace, matching Rust BackendPlugin.
///
/// Calls are exclusive but may run on different host threads (Send, not Sync).
/// Share loaded models/weights; own bindings and execution contexts per
/// instance. Inputs are read-only: isolate SDK mutation/retention inside the
/// implementation. Complete all accesses to borrowed storage before returning,
/// including errors and unwinding. Keep SDK libraries loaded through their
/// final handle's lifetime. Destroy bindings before workspace buffers; failed
/// preparation must be safe to destroy. The engine discards failed workspaces
/// and never retries execution.
class BackendPlugin {
 public:
  /// @brief Release bindings and SDK handles before workspace buffers expire.
  virtual ~BackendPlugin() = default;

  /// @brief Create an unprepared peer sharing loaded resources and model state.
  ///
  /// Do not reload artifacts or copy live bindings, intermediates, or context
  /// state. Construction may overlap another peer's execution: read immutable
  /// metadata or synchronize shared access.
  /// @param[out] peer Empty destination; publish the new instance on success.
  /// @return Success or a backend failure.
  virtual Status new_session(std::unique_ptr<BackendPlugin>* peer) const = 0;

  /// @brief Request the shared graph guard during execution and output copying.
  /// Stable across preparation and all peers. Peer construction must be safe
  /// independently of this guard.
  /// @return true by default; false only when shared-model execution is safe
  /// concurrently.
  virtual bool needs_guard() const noexcept { return true; }

  /// @brief Permit preparation when the engine also has concrete buffers.
  /// @return false by default, selecting dynamic execution even for concrete
  /// model shapes; true to allow a preparation attempt.
  virtual bool can_prepare() const noexcept { return false; }

  /// @brief Report whether this instance has prepared bindings.
  /// @return true only after successful preparation; false for new peers and
  /// by default.
  virtual bool is_prepared() const noexcept { return false; }

  /// @brief Query borrowed prepared-input alignment after initialization.
  /// The engine resolves placement and alignment before run().
  /// @return Address alignment in bytes, or zero to disable borrowing
  /// (default).
  virtual size_t input_alignment() const noexcept { return 0; }

  /// @brief Bind fixed storage and cache wrappers, metadata, and argument
  /// arrays.
  ///
  /// Contents change between calls; layouts and addresses remain fixed until
  /// this instance drops, including failure. Descriptor objects and spans are
  /// temporary: own retained shape/name metadata. Storage is exclusive during
  /// backend operations and available to the engine between completed calls.
  /// Finish work before returning; never mutate input storage.
  /// @param[in] inputs Read-only fixed inputs; allocations may alias.
  /// @param outputs Writable fixed outputs, disjoint from inputs and each
  /// other.
  /// @post is_prepared() is true on success.
  /// @return Success or a backend failure; ErrorCode::kUnsupported by default.
  virtual Status prepare(
      [[maybe_unused]] std::span<const BorrowedTensor> inputs,
      [[maybe_unused]] std::span<const WritableTensor> outputs) {
    return {ErrorCode::kUnsupported, "prepare",
            "backend does not support fixed bindings"};
  }

  /// @brief Query required output alignment, stable across invocations and
  /// peers.
  /// @return Address alignment in bytes; one by default.
  virtual size_t output_alignment() const noexcept { return 1; }

  /// @brief Execute prepared bindings with invocation-scoped boundary inputs.
  ///
  /// Unrouted arguments keep fixed bindings. The engine establishes input
  /// compatibility and placement. Finish work and restore bindings before
  /// returning or unwinding. Never retain caller pointers; no retries.
  /// @param[in] inputs Boundary inputs in Session input order.
  /// @param[in] routes Argument-position to Session-input-index mappings; empty
  /// routes reuse all prepared inputs.
  /// @param outputs Writable, disjoint, compatible invocation output overrides,
  /// alive through completion.
  /// @return Success or a backend failure; ErrorCode::kUnsupported by default.
  virtual Status run([[maybe_unused]] std::span<const NamedTensor> inputs,
                     [[maybe_unused]] std::span<const InputRoute> routes,
                     [[maybe_unused]] std::span<const OutputBinding> outputs) {
    return {ErrorCode::kUnsupported, "execute",
            "backend has no prepared execution path"};
  }

  /// @brief Execute without fixed bindings and publish independent results.
  ///
  /// Return nullopt for supplied destinations and a populated OwnedTensor for
  /// every other output. Results must be ready, contiguous, on the execution
  /// device, and match declared count/order/dtype/rank/fixed dimensions.
  /// Establish SDK guarantees at load time and check unresolved values before
  /// returning. Transfers::copy isolates SDK storage from inputs and mutable
  /// state. Do not retain invocation pointers; publish only on success.
  /// @param[in] inputs Validated read-only inputs, borrowed for this call only.
  /// @param outputs Invocation destinations satisfying run()'s contract.
  /// @param[in,out] transfers Workspace copy helper, exclusively borrowed for
  /// this call only.
  /// @param[out] result Empty destination for one entry per declared output.
  /// @return Success or a backend failure.
  virtual Status run_dynamic(std::span<const BorrowedTensor> inputs,
                             std::span<const OutputBinding> outputs,
                             Transfers& transfers, DynamicOutputs* result) = 0;
};

/// @brief Load a model once; peers share resources via
/// BackendPlugin::new_session(). Build candidates with RAII and publish only on
/// success.
/// @param[in] context Initialization metadata borrowed for this call only.
/// @param[out] plugin Empty destination for the loaded instance.
/// @return Success or a backend failure.
using LoadBackend = Status (*)(const LoadContext& context,
                               std::unique_ptr<BackendPlugin>* plugin);

}  // namespace lrt::native

#endif  // LEAPP_NATIVE_BACKEND_PLUGIN_H_
