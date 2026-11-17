// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_NATIVE_DETAIL_ABI_H_
#define LEAPP_NATIVE_DETAIL_ABI_H_

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>

// Private transport paired with bridge/ffi.rs. No C++ objects cross this ABI.
namespace lrt::native::detail {

struct Error {
  int32_t code;
  const char* message;
  size_t message_size;
  void* owner;
  void (*destroy)(void* owner) noexcept;
};

struct TensorHandle {
  void* value;
  void (*destroy)(void* owner) noexcept;
};

struct Port {
  const char* name;
  size_t name_size;
  DLDataType dtype;
  const int64_t* shape;
  size_t rank;
};

struct LoadArgs {
  const char* path;
  size_t path_size;
  int32_t device;
  void* stream;
  const Port* inputs;
  size_t input_count;
  const Port* outputs;
  size_t output_count;
};

struct NamedTensor {
  const char* name;
  size_t name_size;
  DLTensor tensor;
};

struct InputRoute {
  size_t position;
  size_t boundary;
};

struct OutputBinding {
  size_t position;
  DLTensor tensor;
};

struct Host {
  void* context;
  Error (*copy)(void*, const DLTensor*, TensorHandle*) noexcept;
};

struct Header {
  uint32_t version;
  uint32_t size;
};

struct Api {
  Header header;
  Error (*load)(const LoadArgs*, void**) noexcept;
  void (*destroy)(void* owner) noexcept;
  Error (*new_session)(const void*, void**) noexcept;
  bool (*needs_guard)(const void*) noexcept;
  bool (*can_prepare)(const void*) noexcept;
  bool (*is_prepared)(const void*) noexcept;
  size_t (*input_alignment)(const void*) noexcept;
  size_t (*output_alignment)(const void*) noexcept;
  Error (*prepare)(void*, const DLTensor*, size_t, const DLTensor*,
                   size_t) noexcept;
  Error (*run)(void*, const NamedTensor*, size_t, const InputRoute*, size_t,
               const OutputBinding*, size_t) noexcept;
  Error (*run_dynamic)(void*, const DLTensor*, size_t, const OutputBinding*,
                       size_t, const Host*, void**) noexcept;
  size_t (*output_count)(const void*) noexcept;
  TensorHandle (*take_output)(void*, size_t) noexcept;
  void (*destroy_outputs)(void* outputs) noexcept;
};

}  // namespace lrt::native::detail

#endif  // LEAPP_NATIVE_DETAIL_ABI_H_
