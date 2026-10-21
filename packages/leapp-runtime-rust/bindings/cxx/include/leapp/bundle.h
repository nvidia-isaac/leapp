// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_BUNDLE_H_
#define LEAPP_BUNDLE_H_

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "leapp/data_type.h"
#include "leapp/metadata.h"
#include "leapp/status.h"

namespace leapp {

struct PortRef {
  std::string node;
  std::string port;

  std::string name() const { return node + "/" + port; }
};

struct PortSpec {
  std::string name;
  DataType dtype = DataType::kFloat32;
  std::vector<int64_t> shape;
  std::string tensor_type;
  Metadata metadata;
};

/// @brief Ordered public ports. A missing descriptor preserves an unresolved
/// declaration.
using PortMap = std::vector<std::pair<std::string, std::optional<PortSpec>>>;

/// @brief Unqualified model port names in declaration order, including duplicates.
struct ModelIO {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

/// @brief Shared immutable bundle snapshot and retained C++ inspection values.
/// Getter references survive while any copy retains the snapshot. Concurrent
/// reads are safe; assignment or destruction of the same handle needs
/// synchronization.
class Bundle {
 public:
  Bundle() = default;

  /// @brief Open into an empty destination. Failure preserves *bundle.
  static Status Open(const std::filesystem::path& path, Bundle* bundle);

  bool is_valid() const { return data_ != nullptr; }

  const std::filesystem::path& yaml_path() const;
  const std::vector<std::string>& model_names() const;

  /// @brief Borrow input/output names, including internal ports. An absent model
  /// returns nullptr; a model with no ports has empty lists. The pointer survives
  /// while any Bundle copy retains the snapshot.
  const ModelIO* model_io(std::string_view node) const;

  const PortMap& pipeline_inputs() const;
  const PortMap& pipeline_outputs() const;
  const Metadata& pipeline_configs() const;
  const Metadata& system_information() const;
  const Metadata& metadata() const;
  const Metadata& pipeline_metadata() const;

  /// @brief Borrow model metadata or parameters; absent models return nullptr.
  const Metadata* model_parameters(std::string_view node) const;
  const Metadata* model_metadata(std::string_view node) const;

  /// @brief Borrow the first matching port descriptor, or nullptr if absent.
  const PortSpec* input_info(std::string_view node,
                             std::string_view port) const;
  const PortSpec* output_info(std::string_view node,
                              std::string_view port) const;

 private:
  friend class Session;
  struct Data;
  std::shared_ptr<Data> data_;
};

}  // namespace leapp

#endif  // LEAPP_BUNDLE_H_
