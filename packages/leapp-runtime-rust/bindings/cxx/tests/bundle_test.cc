// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "leapp/bundle.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace {
void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}
}  // namespace

int main(int argc, char** argv) {
  Check(argc == 2, "expected the pipeline fixture path");
  const auto nonce =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto directory = std::filesystem::temp_directory_path() /
                         ("leapp-bundle-" + std::to_string(nonce));
  Check(std::filesystem::create_directory(directory),
        "create fixture directory");
  const auto path = directory / "pipeline.yaml";
  std::filesystem::copy_file(argv[1], path);
  std::ofstream(directory / "model.bin", std::ios::binary) << "abc";
  std::ofstream(path, std::ios::app)
      << "sensor: {axes: [x, y]}\n"
         "sequence_keys: {[x, y]: coordinates}\n"
         "unsigned: 18446744073709551615\n";

  leapp::Bundle empty;
  Check(!empty.is_valid() && empty.model_names().empty(), "empty handle");
  Check(empty.input_info("missing", "port") == nullptr, "empty descriptor");
  Check(!leapp::Bundle::Open(path, nullptr).ok(), "null destination");
  auto status = leapp::Bundle::Open(directory / "absent.yaml", &empty);
  Check(status.code() == leapp::ErrorCode::kNotFound, "missing YAML category");
  Check(status.phase() == "open" && !status.message().empty(),
        "error diagnostics");
  Check(!empty.is_valid(), "failure preserves empty destination");

  leapp::Bundle bundle;
  Check(leapp::Bundle::Open(path, &bundle).ok(), "load opaque artifacts");
  Check(bundle.yaml_path() == path, "absolute path");
  Check(bundle.model_names() ==
            std::vector<std::string>({"z_observation", "a_policy"}),
        "model declaration order");
  Check(bundle.pipeline_inputs().size() == 1, "pipeline inputs");
  Check(bundle.pipeline_outputs().size() == 3, "pipeline outputs");
  Check(bundle.pipeline_outputs()[0].first == "a_policy/next_state",
        "output order");
  const auto* port = bundle.output_info("a_policy", "actions");
  Check(port != nullptr && port->dtype == leapp::DataType::kFloat32,
        "output dtype");
  Check(port->shape == std::vector<int64_t>({3}), "output shape");
  Check(bundle.input_info("absent", "port") == nullptr, "absent descriptor");
  Check(bundle.model_parameters("absent") == nullptr, "absent parameters");
  const auto* frequency = bundle.pipeline_configs().Find("frequency");
  Check(frequency && std::get<int64_t>(frequency->value) == 50, "typed config");
  const auto* sensor = bundle.metadata().Find("sensor");
  Check(sensor && sensor->Find("axes"), "nested mapping metadata");
  const auto* mapping = bundle.metadata().Find("sequence_keys");
  Check(mapping != nullptr, "mapping metadata");
  const auto& entries = std::get<leapp::Metadata::Mapping>(mapping->value);
  Check(std::get<leapp::Metadata::Sequence>(entries[0].key.value).size() == 2,
        "non-string mapping key");
  const auto* unsigned_value = bundle.metadata().Find("unsigned");
  Check(unsigned_value &&
            std::get<uint64_t>(unsigned_value->value) == UINT64_MAX,
        "unsigned metadata");

  Check(!leapp::Bundle::Open(path, &bundle).ok(),
        "reject occupied destination");
  Check(bundle.output_info("a_policy", "actions") == port,
        "failed open preserves snapshot");
  auto retained = bundle;
  bundle = {};
  std::filesystem::remove(path);
  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([retained, port] {
      for (int j = 0; j < 100; ++j) {
        Check(retained.output_info("a_policy", "actions") == port,
              "shared snapshot lifetime");
        Check(retained.input_info("z_observation", "joint_pos") != nullptr,
              "concurrent descriptor lookup");
      }
    });
  }
  for (auto& reader : readers) {
    reader.join();
  }
  std::filesystem::remove_all(directory);
}
