// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <dlfcn.h>

#include "leapp/runtime.h"

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  void* plugin = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!plugin) return 1;
  const auto factory =
      reinterpret_cast<const void* (*)()>(dlsym(plugin, "leapp_native_example_v1"));
  const bool loaded = factory && factory();
  const bool closed = dlclose(plugin) == 0;
  lrt::Bundle bundle;
  return !loaded || !closed || lrt::Bundle::Open("missing.yaml", &bundle).ok();
}
