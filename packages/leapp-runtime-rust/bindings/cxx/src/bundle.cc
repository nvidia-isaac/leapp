// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "leapp/bundle.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "leapp-runtime-cxx/src/bridge.rs.h"

namespace lrt {
namespace {
namespace wire = detail;

// Keep casts across the private bridge checked when either enum changes.
static_assert(static_cast<uint8_t>(DataType::kBool) ==
              static_cast<uint8_t>(wire::Dtype::Bool));
static_assert(static_cast<uint8_t>(DataType::kUint8) ==
              static_cast<uint8_t>(wire::Dtype::Uint8));
static_assert(static_cast<uint8_t>(DataType::kUint16) ==
              static_cast<uint8_t>(wire::Dtype::Uint16));
static_assert(static_cast<uint8_t>(DataType::kUint32) ==
              static_cast<uint8_t>(wire::Dtype::Uint32));
static_assert(static_cast<uint8_t>(DataType::kUint64) ==
              static_cast<uint8_t>(wire::Dtype::Uint64));
static_assert(static_cast<uint8_t>(DataType::kInt8) ==
              static_cast<uint8_t>(wire::Dtype::Int8));
static_assert(static_cast<uint8_t>(DataType::kInt16) ==
              static_cast<uint8_t>(wire::Dtype::Int16));
static_assert(static_cast<uint8_t>(DataType::kInt32) ==
              static_cast<uint8_t>(wire::Dtype::Int32));
static_assert(static_cast<uint8_t>(DataType::kInt64) ==
              static_cast<uint8_t>(wire::Dtype::Int64));
static_assert(static_cast<uint8_t>(DataType::kFloat16) ==
              static_cast<uint8_t>(wire::Dtype::Float16));
static_assert(static_cast<uint8_t>(DataType::kBfloat16) ==
              static_cast<uint8_t>(wire::Dtype::Bfloat16));
static_assert(static_cast<uint8_t>(DataType::kFloat32) ==
              static_cast<uint8_t>(wire::Dtype::Float32));
static_assert(static_cast<uint8_t>(DataType::kFloat64) ==
              static_cast<uint8_t>(wire::Dtype::Float64));

std::string String(const rust::String& value) {
  return {value.data(), value.size()};
}

rust::Slice<const uint8_t> Bytes(std::string_view value) {
  return {reinterpret_cast<const uint8_t*>(value.data()), value.size()};
}

Status Convert(const wire::StatusData& status) {
  return {static_cast<ErrorCode>(status.code), String(status.phase),
          String(status.message)};
}

Status Invalid(std::string message) {
  return {ErrorCode::kInvalidArgument, "binding", std::move(message)};
}

const Metadata& EmptyMetadata() {
  static const Metadata empty;
  return empty;
}

// Decode the typed preorder traversal without reparsing YAML or serializing
// text.
Metadata ReadMetadata(const rust::Vec<wire::MetadataNode>& nodes,
                      size_t* index) {
  const auto& node = nodes[(*index)++];
  using Kind = wire::MetadataKind;
  switch (node.kind) {
    case Kind::Null:
      return {};
    case Kind::Bool:
      return {node.boolean};
    case Kind::Int:
      return {node.integer};
    case Kind::Uint:
      return {node.unsigned_integer};
    case Kind::Float:
      return {node.number};
    case Kind::String:
      return {String(node.text)};
    case Kind::Sequence: {
      Metadata::Sequence items;
      items.reserve(node.children);
      for (size_t i = 0; i < node.children; ++i) {
        items.push_back(ReadMetadata(nodes, index));
      }
      return {std::move(items)};
    }
    case Kind::Mapping: {
      Metadata::Mapping items;
      items.reserve(node.children);
      for (size_t i = 0; i < node.children; ++i) {
        auto key = ReadMetadata(nodes, index);
        auto value = ReadMetadata(nodes, index);
        items.push_back({std::move(key), std::move(value)});
      }
      return {std::move(items)};
    }
  }
  // An unknown kind violates the private generated bridge contract.
  std::abort();
}

Metadata ReadMetadata(const rust::Vec<wire::MetadataNode>& nodes) {
  size_t index = 0;
  return nodes.empty() ? Metadata{} : ReadMetadata(nodes, &index);
}

Metadata Field(const wire::BundleHandle& bundle, wire::MetadataField field,
               std::string_view name = {}) {
  return ReadMetadata(wire::bundle_metadata(bundle, field, Bytes(name)).nodes);
}

}  // namespace

const Metadata* Metadata::Find(std::string_view key) const {
  const auto* mapping = std::get_if<Mapping>(&value);
  if (mapping == nullptr) {
    return nullptr;
  }
  for (const auto& entry : *mapping) {
    const auto* name = std::get_if<std::string>(&entry.key.value);
    if (name != nullptr && *name == key) {
      return &entry.value;
    }
  }
  return nullptr;
}

struct Bundle::Data {
  explicit Data(rust::Box<wire::BundleHandle> value)
      : handle(std::move(value)) {
    const auto bytes = wire::bundle_path(*handle);
    path = std::string(bytes.begin(), bytes.end());
    for (const auto& name : wire::bundle_models(*handle)) {
      names.push_back(String(name));
    }
    metadata = Field(*handle, wire::MetadataField::Root);
    system = Field(*handle, wire::MetadataField::System);
    configs = Field(*handle, wire::MetadataField::Configs);
    pipeline = Field(*handle, wire::MetadataField::Pipeline);
    for (const auto& name : names) {
      parameters.emplace(name,
                         Field(*handle, wire::MetadataField::Parameters, name));
      models.emplace(name, Field(*handle, wire::MetadataField::Model, name));
      const auto io = wire::bundle_model_io(*handle, Bytes(name));
      if (io.present) {
        ModelIO ports;
        for (const auto& input : io.inputs) {
          ports.inputs.push_back(String(input));
        }
        for (const auto& output : io.outputs) {
          ports.outputs.push_back(String(output));
        }
        model_ports.emplace(name, std::move(ports));
      }
    }
    inputs = Ports(true);
    outputs = Ports(false);
  }

  const PortSpec* Port(std::string_view node, std::string_view port,
                       bool input) const {
    std::lock_guard lock(mutex);
    auto key = std::make_tuple(std::string(node), std::string(port), input);
    if (auto it = ports.find(key); it != ports.end()) {
      return it->second ? &*it->second : nullptr;
    }
    const auto value =
        wire::bundle_port(*handle, Bytes(node), Bytes(port), input);
    std::optional<PortSpec> spec;
    if (value.present) {
      spec = PortSpec{String(value.name),
                      static_cast<DataType>(value.dtype),
                      {value.shape.begin(), value.shape.end()},
                      String(value.tensor_type),
                      ReadMetadata(value.metadata)};
    }
    const auto it = ports.emplace(std::move(key), std::move(spec)).first;
    return it->second ? &*it->second : nullptr;
  }

  PortMap Ports(bool input) const {
    PortMap result;
    for (const auto& p : wire::bundle_ports(*handle, input)) {
      const auto node = String(p.node);
      const auto port = String(p.port);
      const auto name = node + "/" + port;
      if (std::any_of(result.begin(), result.end(),
                      [&](const auto& entry) { return entry.first == name; })) {
        continue;
      }
      const auto* info = Port(node, port, input);
      result.emplace_back(name, info == nullptr
                                    ? std::nullopt
                                    : std::optional<PortSpec>(*info));
    }
    return result;
  }

  rust::Box<wire::BundleHandle> handle;
  std::filesystem::path path;
  std::vector<std::string> names;
  PortMap inputs;
  PortMap outputs;
  Metadata metadata;
  Metadata system;
  Metadata configs;
  Metadata pipeline;
  std::map<std::string, Metadata, std::less<>> parameters;
  std::map<std::string, Metadata, std::less<>> models;
  std::map<std::string, ModelIO, std::less<>> model_ports;
  // Map nodes never move; returned descriptor addresses survive later lookups.
  mutable std::mutex mutex;
  mutable std::map<std::tuple<std::string, std::string, bool>,
                   std::optional<PortSpec>>
      ports;
};

Status Bundle::Open(const std::filesystem::path& path, Bundle* bundle) {
  if (bundle == nullptr || bundle->is_valid()) {
    return Invalid("bundle destination must be non-null and empty");
  }
  auto handle = wire::bundle_new();
  auto status = Convert(wire::bundle_open(Bytes(path.native()), *handle));
  if (!status.ok()) {
    return status;
  }
  bundle->data_ = std::make_shared<Data>(std::move(handle));
  return {};
}

const std::filesystem::path& Bundle::yaml_path() const {
  static const std::filesystem::path empty;
  return data_ ? data_->path : empty;
}

const std::vector<std::string>& Bundle::model_names() const {
  static const std::vector<std::string> empty;
  return data_ ? data_->names : empty;
}

const ModelIO* Bundle::model_io(std::string_view node) const {
  if (!data_) {
    return nullptr;
  }
  const auto it = data_->model_ports.find(node);
  return it == data_->model_ports.end() ? nullptr : &it->second;
}

const PortMap& Bundle::pipeline_inputs() const {
  static const PortMap empty;
  return data_ ? data_->inputs : empty;
}

const PortMap& Bundle::pipeline_outputs() const {
  static const PortMap empty;
  return data_ ? data_->outputs : empty;
}

const Metadata& Bundle::metadata() const {
  return data_ ? data_->metadata : EmptyMetadata();
}

const Metadata& Bundle::system_information() const {
  return data_ ? data_->system : EmptyMetadata();
}

const Metadata& Bundle::pipeline_configs() const {
  return data_ ? data_->configs : EmptyMetadata();
}

const Metadata& Bundle::pipeline_metadata() const {
  return data_ ? data_->pipeline : EmptyMetadata();
}

const Metadata* Bundle::model_parameters(std::string_view node) const {
  if (!data_) {
    return nullptr;
  }
  const auto it = data_->parameters.find(node);
  return it == data_->parameters.end() ? nullptr : &it->second;
}

const Metadata* Bundle::model_metadata(std::string_view node) const {
  if (!data_) {
    return nullptr;
  }
  const auto it = data_->models.find(node);
  return it == data_->models.end() ? nullptr : &it->second;
}

const PortSpec* Bundle::input_info(std::string_view node,
                                   std::string_view port) const {
  return data_ ? data_->Port(node, port, true) : nullptr;
}

const PortSpec* Bundle::output_info(std::string_view node,
                                    std::string_view port) const {
  return data_ ? data_->Port(node, port, false) : nullptr;
}

}  // namespace lrt
