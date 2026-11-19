// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "leapp/runtime.h"

#include <dlfcn.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << std::endl;
    std::abort();
  }
}
}  // namespace

int main(int argc, char** argv) {
  Check(argc == 2, "expected plugin path");
  // Resolve a consumer-defined plugin built with a different STL ABI.
  void* plugin = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!plugin) std::cerr << dlerror() << '\n';
  Check(plugin != nullptr, "load consumer plugin");
  using Factory = const void* (*)();
  const auto factory =
      reinterpret_cast<Factory>(dlsym(plugin, "leapp_native_example_v1"));
  Check(factory != nullptr && factory() != nullptr, "resolve adapter factory");
  Check(dlclose(plugin) == 0, "unload plugin");

  const auto path =
      std::filesystem::temp_directory_path() / "leapp-bazel-cxx.yaml";
  std::ofstream(path) << "models: {}\npipeline: {inputs: {}, outputs: {}}\n"
                         "system information: {}\n";
  lrt::Bundle bundle;
  Check(lrt::Bundle::Open(path, &bundle).ok(), "inspect bundle");
  std::filesystem::remove(path);
  Check(bundle.model_names().empty(), "empty model names");
  lrt::Session session;
  const auto status = lrt::Session::Open(bundle, {}, &session);
  Check(status.code() == lrt::ErrorCode::kInvalidArgument &&
            status.phase() == "convert",
        "reject empty graph before loading SDKs");
}
