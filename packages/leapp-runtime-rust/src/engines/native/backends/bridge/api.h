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

struct InputRoute {
  size_t position;
  size_t boundary;  // Session input index; Rust resolves its current View data.
};

struct OutputBinding {
  size_t position;
  View view;
};

// Fallible calls return null or an owned diagnostic released by free_error.
// Prepared workspace storage lives until destroy_call. Invocation input/output
// bindings and metadata live through run only; returned views live until
// destroy_outputs. Complete work and restore prepared bindings before
// returning, including errors. Destruction order: calls/outputs, session,
// model, library.
struct Api {
  // Optional caller-owned CUDA stream; null requests internal streams. Retain
  // it through model/session destruction. Peers inherit the same selection.
  char* (*load_model)(const char*, int32_t, void*, const Port*, size_t, const Port*,
                      size_t, void**) noexcept;
  void (*destroy_model)(void*) noexcept;
  bool (*requires_dynamic)(const void*) noexcept;
  char* (*create_session)(void*, void**) noexcept;
  void (*destroy_session)(void*) noexcept;
  char* (*prepare)(void*, const View*, size_t, const View*, size_t,
                   void**) noexcept;
  void (*destroy_call)(void*) noexcept;
  size_t (*input_alignment)(const void*) noexcept;
  char* (*run)(void*, const View*, size_t, const InputRoute*, size_t,
               const OutputBinding*, size_t) noexcept;
  char* (*run_dynamic)(void*, const View*, size_t, const OutputBinding*, size_t,
                       void**) noexcept;
  void (*destroy_outputs)(void*) noexcept;
  const View* (*views)(const void*, size_t*) noexcept;
  void (*free_error)(char*) noexcept;
};

}  // namespace leapp_native::bridge
