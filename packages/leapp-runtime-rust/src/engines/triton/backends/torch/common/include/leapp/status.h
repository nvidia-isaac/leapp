// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file
/// @brief Owned error codes and diagnostics returned by fallible operations.

#ifndef COMMON_INCLUDE_LEAPP_STATUS_H_
#define COMMON_INCLUDE_LEAPP_STATUS_H_

#include <string>
#include <utility>

namespace leapp {

/// @brief Failure categories shared by the C++ and C interfaces.
enum class ErrorCode {
  kOk,               ///< Operation succeeded.
  kInvalidArgument,  ///< Invalid arguments or bundle contents.
  kNotFound,         ///< A required file or resource does not exist.
  kUnsupported,      ///< The requested bundle feature is not supported.
  kIoError,          ///< File access or another I/O operation failed.
  kInternal,         ///< An unexpected implementation or dependency failure.
};

/// @brief An operation result that owns its phase and diagnostic strings.
///
/// Default construction represents success with empty diagnostics. References
/// returned by accessors remain valid while the Status is alive and unchanged.
class Status {
 public:
  Status() = default;

  Status(ErrorCode code, std::string phase, std::string message)
      : code_(code), phase_(std::move(phase)), message_(std::move(message)) {}

  bool ok() const { return code_ == ErrorCode::kOk; }

  ErrorCode code() const { return code_; }

  const std::string& phase() const { return phase_; }

  const std::string& message() const { return message_; }

 private:
  ErrorCode code_ = ErrorCode::kOk;
  std::string phase_;
  std::string message_;
};

}  // namespace leapp

#endif  // COMMON_INCLUDE_LEAPP_STATUS_H_
