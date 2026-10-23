// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/model_config.h"

#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "src/backend_utils.h"

namespace leapp::internal {

Status ReadModelConfig(TRITONBACKEND_Model* model, ModelConfig* state) {
  constexpr uint32_t kConfigVersion = 1;

  TRITONSERVER_Message* message = nullptr;
  auto status = TakeBackendError(
      TRITONBACKEND_ModelConfig(model, kConfigVersion, &message));
  if (!status.ok()) {
    return status;
  }
  const std::unique_ptr<TRITONSERVER_Message,
                        decltype(&TRITONSERVER_MessageDelete)>
      owner(message, TRITONSERVER_MessageDelete);
  const char* json = nullptr;
  size_t size = 0;
  status = TakeBackendError(
      TRITONSERVER_MessageSerializeToJson(message, &json, &size));
  if (!status.ok()) {
    return status;
  }
  const auto config = nlohmann::json::parse(json, json + size);
  if (config.value("max_batch_size", 0) != 0 ||
      config.value("model_transaction_policy", nlohmann::json::object())
          .value("decoupled", false)) {
    return {ErrorCode::kUnsupported, "backend",
            "models require max_batch_size: 0 and coupled responses"};
  }
  const auto read_ports = [&](const char* field,
                              std::vector<ModelPort>* ports) {
    for (const auto& value : config.value(field, nlohmann::json::array())) {
      const auto dtype = value.at("data_type").get<std::string>();
      ports->push_back({value.at("name").get<std::string>(),
                        TRITONSERVER_StringToDataType(
                            dtype.substr(std::string("TYPE_").size()).c_str()),
                        value.at("dims").get<std::vector<int64_t>>()});
    }
  };
  read_ports("input", &state->inputs);
  read_ports("output", &state->outputs);
  TRITONBACKEND_ArtifactType artifact_type{};
  const char* repository = nullptr;
  uint64_t version = 0;
  status = TakeBackendError(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &repository));
  if (!status.ok()) {
    return status;
  }
  if (artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    return {ErrorCode::kUnsupported, "backend",
            "models require a filesystem repository"};
  }
  status = TakeBackendError(TRITONBACKEND_ModelVersion(model, &version));
  if (!status.ok()) {
    return status;
  }
  state->artifact_path = std::filesystem::path(repository) /
                         std::to_string(version) /
                         config.at("default_model_filename").get<std::string>();
  return {};
}

}  // namespace leapp::internal
