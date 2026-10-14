// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_STATUS_H_
#define LEAPP_STATUS_H_

#include <string>
#include <utility>

namespace leapp {

/// @brief Failure categories shared with the Rust runtime.
enum class ErrorCode {
  kOk,
  kInvalidArgument,
  kNotFound,
  kUnsupported,
  kIoError,
  kInternal
};

/// @brief Owned operation status; default construction represents success.
class [[nodiscard]] Status {
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

#endif  // LEAPP_STATUS_H_
