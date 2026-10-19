// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAPP_METADATA_H_
#define LEAPP_METADATA_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace leapp {

/// @brief Detached typed metadata, preserving ordered mappings and non-string
/// keys.
struct Metadata {
  struct Entry;

  using Sequence = std::vector<Metadata>;
  using Mapping = std::vector<Entry>;
  using Value =
      std::variant<std::monostate, bool, int64_t, uint64_t, double, std::string,
                   Sequence, Mapping>;

  Value value;  ///< Defaults to null.

  /// @brief Borrow a string-keyed mapping value; absent keys return nullptr.
  const Metadata* Find(std::string_view key) const;
};

struct Metadata::Entry {
  Metadata key;
  Metadata value;
};

}  // namespace leapp

#endif  // LEAPP_METADATA_H_
