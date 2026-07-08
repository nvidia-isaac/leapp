#include "runtime_metadata.h"

#include <regex>
#include <stdexcept>

namespace leapp::warp_runtime {
namespace {

std::string ExtractString(const std::string& object, const std::string& key, bool required = true) {
    const std::regex pattern("\\\"" + key + "\\\":(?:null|\\\"([^\\\"]*)\\\")");
    std::smatch match;
    if (!std::regex_search(object, match, pattern)) {
        if (required) {
            throw std::runtime_error("runtime_metadata missing string key: " + key);
        }
        return "";
    }
    return match.size() > 1 ? match[1].str() : "";
}

long long ExtractInt(const std::string& object, const std::string& key, long long fallback = 0, bool required = true) {
    const std::regex pattern("\\\"" + key + "\\\":(-?[0-9]+)");
    std::smatch match;
    if (!std::regex_search(object, match, pattern)) {
        if (required) {
            throw std::runtime_error("runtime_metadata missing integer key: " + key);
        }
        return fallback;
    }
    return std::stoll(match[1].str());
}

bool ExtractBool(const std::string& object, const std::string& key, bool fallback = false, bool required = true) {
    const std::regex pattern("\\\"" + key + "\\\":(true|false)");
    std::smatch match;
    if (!std::regex_search(object, match, pattern)) {
        if (required) {
            throw std::runtime_error("runtime_metadata missing bool key: " + key);
        }
        return fallback;
    }
    return match[1].str() == "true";
}

std::vector<std::int64_t> ExtractShape(const std::string& object) {
    const std::regex pattern("\\\"shape\\\":\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(object, match, pattern)) {
        throw std::runtime_error("runtime_metadata missing shape");
    }
    std::vector<std::int64_t> shape;
    const std::string body = match[1].str();
    std::regex number("-?[0-9]+");
    for (std::sregex_iterator it(body.begin(), body.end(), number), end; it != end; ++it) {
        shape.push_back(std::stoll((*it)[0].str()));
    }
    return shape;
}

std::vector<std::string> ExtractObjectArray(const std::string& json, const std::string& key) {
    const std::string marker = "\"" + key + "\":[";
    const std::size_t start_marker = json.find(marker);
    if (start_marker == std::string::npos) {
        throw std::runtime_error("runtime_metadata missing array: " + key);
    }
    std::size_t pos = start_marker + marker.size();
    int array_depth = 1;
    int object_depth = 0;
    std::size_t object_start = std::string::npos;
    std::vector<std::string> objects;

    for (; pos < json.size(); ++pos) {
        const char c = json[pos];
        if (c == '[' && object_depth == 0) {
            ++array_depth;
        } else if (c == ']' && object_depth == 0) {
            --array_depth;
            if (array_depth == 0) {
                break;
            }
        } else if (c == '{') {
            if (object_depth == 0) {
                object_start = pos;
            }
            ++object_depth;
        } else if (c == '}') {
            --object_depth;
            if (object_depth == 0 && object_start != std::string::npos) {
                objects.push_back(json.substr(object_start, pos - object_start + 1));
                object_start = std::string::npos;
            }
        }
    }
    if (array_depth != 0 || object_depth != 0) {
        throw std::runtime_error("runtime_metadata has malformed array: " + key);
    }
    return objects;
}

}  // namespace

RuntimeMetadata ParseRuntimeMetadata(const std::string& json) {
    RuntimeMetadata metadata;
    metadata.schema_version = static_cast<int>(ExtractInt(json, "schema_version"));
    if (metadata.schema_version != 1) {
        throw std::runtime_error("Unsupported runtime_metadata schema_version: " + std::to_string(metadata.schema_version));
    }
    metadata.device_kind = ExtractString(json, "device_kind", false);
    metadata.device_index = static_cast<int>(ExtractInt(json, "device_index", 0, false));
    metadata.bundle_num_bytes = static_cast<std::size_t>(ExtractInt(json, "num_bytes", 0, false));
    metadata.bundle_sha256 = ExtractString(json, "sha256", false);

    for (const std::string& object : ExtractObjectArray(json, "inputs")) {
        InputSpec spec;
        spec.logical_index = static_cast<int>(ExtractInt(object, "logical_index"));
        spec.param_name = ExtractString(object, "param_name");
        spec.dtype = ElementTypeFromName(ExtractString(object, "dtype"));
        spec.shape = ExtractShape(object);
        spec.num_bytes = static_cast<std::size_t>(ExtractInt(object, "num_bytes"));
        if (spec.dtype == ElementType::Unknown) {
            throw std::runtime_error("Unsupported Warp input dtype: " + ElementTypeName(spec.dtype));
        }
        metadata.inputs.push_back(std::move(spec));
    }

    for (const std::string& object : ExtractObjectArray(json, "outputs")) {
        OutputSpec spec;
        spec.logical_index = static_cast<int>(ExtractInt(object, "logical_index"));
        spec.mask = ExtractBool(object, "mask");
        spec.param_name = ExtractString(object, "param_name", false);
        spec.dtype = ElementTypeFromName(ExtractString(object, "dtype"));
        spec.shape = ExtractShape(object);
        spec.num_bytes = static_cast<std::size_t>(ExtractInt(object, "num_bytes"));
        if (spec.mask && spec.param_name.empty()) {
            throw std::runtime_error("Masked Warp output is missing param_name");
        }
        if (spec.dtype == ElementType::Unknown) {
            throw std::runtime_error("Unsupported Warp output dtype");
        }
        metadata.outputs.push_back(std::move(spec));
    }

    return metadata;
}

}  // namespace leapp::warp_runtime
