// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_DATA_TYPE_H_
#define LEAPP_DATA_TYPE_H_

#include <cstdint>

namespace lrt {

/// @brief Fixed-width scalar types; half and Boolean values use packed bytes.
enum class DataType : uint8_t {
  kBool,
  kUint8,
  kUint16,
  kUint32,
  kUint64,
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kFloat16,
  kBfloat16,
  kFloat32,
  kFloat64
};

}  // namespace lrt

#endif  // LEAPP_DATA_TYPE_H_
