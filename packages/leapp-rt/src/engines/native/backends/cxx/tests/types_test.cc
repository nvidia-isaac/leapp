// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <limits>

#include "leapp/native/backend_plugin.h"

void check(bool condition) {
  if (!condition) {
    std::abort();
  }
}

int main() {
  namespace native = lrt::native;

  struct Case {
    lrt::DataType type;
    uint8_t code;
    uint8_t bits;
  };

  const Case cases[] = {
      {lrt::DataType::kBool, kDLBool, 8},
      {lrt::DataType::kUint8, kDLUInt, 8},
      {lrt::DataType::kUint16, kDLUInt, 16},
      {lrt::DataType::kUint32, kDLUInt, 32},
      {lrt::DataType::kUint64, kDLUInt, 64},
      {lrt::DataType::kInt8, kDLInt, 8},
      {lrt::DataType::kInt16, kDLInt, 16},
      {lrt::DataType::kInt32, kDLInt, 32},
      {lrt::DataType::kInt64, kDLInt, 64},
      {lrt::DataType::kFloat16, kDLFloat, 16},
      {lrt::DataType::kBfloat16, kDLBfloat, 16},
      {lrt::DataType::kFloat32, kDLFloat, 32},
      {lrt::DataType::kFloat64, kDLFloat, 64},
  };
  for (const auto& entry : cases) {
    const auto actual = native::dl_type(entry.type);
    check(actual.code == entry.code && actual.bits == entry.bits &&
          actual.lanes == 1);
    check(native::data_type({entry.code, entry.bits, 1}) == entry.type);
  }
  int64_t dimensions[] = {0, std::numeric_limits<int64_t>::max()};
  DLTensor tensor{
      nullptr,    {kDLCPU, 0}, 2, native::dl_type(lrt::DataType::kFloat32),
      dimensions, nullptr,     0};
  check(native::BorrowedTensor(tensor).bytes() == 0);
  tensor.ndim = 0;
  check(native::BorrowedTensor(tensor).bytes() == 4);
  tensor.ndim = 2;
  dimensions[0] = 8;
  bool failed = false;
  try {
    native::BorrowedTensor(tensor).bytes();
  } catch (const native::BackendError& error) {
    failed = error.status().code() == lrt::ErrorCode::kInvalidArgument;
  }
  check(failed);
}
