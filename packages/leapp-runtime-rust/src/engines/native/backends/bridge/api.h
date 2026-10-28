// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace leapp_native::bridge {

// Private native backend transport, paired with ffi.rs. No SDK/STL objects or
// exceptions cross this boundary. Metadata is borrowed for the call only.
struct Port {
  const char* name;
  size_t name_size;
  uint8_t dtype;
  const int64_t* shape;
  size_t rank;
};

struct View {
  Port port;
  uintptr_t data;
  size_t bytes;
  int32_t device;  // CUDA ordinal; -1 for CPU tensors.
};

// Fallible calls return null or an owned diagnostic released by free_error.
// All work finishes before return, including errors. A prepared call borrows
// fixed workspace buffers until destroy_call. Output views live until
// destroy_outputs; Rust copies them before releasing their owner.
// Destruction order: calls/outputs, session, model, library. Access is
// exclusive per session/call. Loaded models may have multiple sessions. TensorRT
// engines are shared read-only; Torch runners require the Rust graph guard.
struct Api {
  uint32_t version;
  char* (*load_model)(const char*, int32_t, const Port*, size_t, const Port*,
                      size_t, void**) noexcept;
  void (*destroy_model)(void*) noexcept;
  bool (*requires_dynamic)(const void*) noexcept;
  char* (*create_session)(void*, void**) noexcept;
  void (*destroy_session)(void*) noexcept;
  char* (*prepare)(void*, const View*, size_t, const View*, size_t,
                   void**) noexcept;
  void (*destroy_call)(void*) noexcept;
  char* (*run)(void*) noexcept;
  char* (*run_dynamic)(void*, const View*, size_t, void**) noexcept;
  void (*destroy_outputs)(void*) noexcept;
  const View* (*views)(const void*, size_t*) noexcept;
  void (*free_error)(char*) noexcept;
};

}  // namespace leapp_native::bridge
