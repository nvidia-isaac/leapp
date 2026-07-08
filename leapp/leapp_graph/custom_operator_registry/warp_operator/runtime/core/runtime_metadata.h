#pragma once

#include "tensor_view.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace leapp::warp_runtime {

struct InputSpec {
    int logical_index = 0;
    std::string param_name;
    ElementType dtype = ElementType::Unknown;
    std::vector<std::int64_t> shape;
    std::size_t num_bytes = 0;
};

struct OutputSpec {
    int logical_index = 0;
    bool mask = true;
    std::string param_name;
    ElementType dtype = ElementType::Unknown;
    std::vector<std::int64_t> shape;
    std::size_t num_bytes = 0;
    double constant_fill = 0.0;
};

struct RuntimeMetadata {
    int schema_version = 0;
    std::string device_kind = "cuda";
    int device_index = 0;
    std::vector<InputSpec> inputs;
    std::vector<OutputSpec> outputs;
    std::size_t bundle_num_bytes = 0;
    std::string bundle_sha256;
};

RuntimeMetadata ParseRuntimeMetadata(const std::string& json);

}  // namespace leapp::warp_runtime
