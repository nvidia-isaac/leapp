// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_NATIVE_EXPORT_H_
#define LEAPP_NATIVE_EXPORT_H_

#include "leapp/native/backend_plugin.h"

namespace lrt::native::detail {

Error load(LoadBackend, const LoadArgs*, void**) noexcept;
Api make_api(decltype(Api::load)) noexcept;

template <LoadBackend Loader>
const Api* export_backend() noexcept {
  static const Api api =
      make_api([](const LoadArgs* args, void** value) noexcept {
        return load(Loader, args, value);
      });
  return &api;
}

}  // namespace lrt::native::detail

#define LEAPP_NATIVE_ENTRY_IMPL(name) leapp_native_##name##_v1
#define LEAPP_NATIVE_ENTRY(name) LEAPP_NATIVE_ENTRY_IMPL(name)

/// Export the generic adapter for one backend. name matches its registered
/// library-directory name; loader has the LoadBackend signature. Link the
/// lrt::native_backend CMake target; implement no raw ABI callbacks.
#define LEAPP_NATIVE_BACKEND(name, loader)                  \
  extern "C" __attribute__((visibility("default")))         \
  const ::lrt::native::detail::Api*                         \
  LEAPP_NATIVE_ENTRY(name)() noexcept {                     \
    return ::lrt::native::detail::export_backend<loader>(); \
  }

#endif  // LEAPP_NATIVE_EXPORT_H_
