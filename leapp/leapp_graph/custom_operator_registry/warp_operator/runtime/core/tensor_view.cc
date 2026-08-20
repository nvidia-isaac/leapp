#include "tensor_view.h"

#include <stdexcept>

namespace leapp::warp_runtime {

std::size_t ElementSize(ElementType dtype) {
    switch (dtype) {
        case ElementType::Bool:
        case ElementType::UInt8:
        case ElementType::Int8:
            return 1;
        case ElementType::Int16:
        case ElementType::Float16:
        case ElementType::BFloat16:
            return 2;
        case ElementType::Int32:
        case ElementType::Float32:
            return 4;
        case ElementType::Int64:
        case ElementType::Float64:
            return 8;
        default:
            return 0;
    }
}

ElementType ElementTypeFromName(const std::string& name) {
    if (name == "bool") return ElementType::Bool;
    if (name == "uint8") return ElementType::UInt8;
    if (name == "int8") return ElementType::Int8;
    if (name == "int16") return ElementType::Int16;
    if (name == "int32") return ElementType::Int32;
    if (name == "int64") return ElementType::Int64;
    if (name == "float16") return ElementType::Float16;
    if (name == "bfloat16") return ElementType::BFloat16;
    if (name == "float32") return ElementType::Float32;
    if (name == "float64") return ElementType::Float64;
    return ElementType::Unknown;
}

std::string ElementTypeName(ElementType dtype) {
    switch (dtype) {
        case ElementType::Bool: return "bool";
        case ElementType::UInt8: return "uint8";
        case ElementType::Int8: return "int8";
        case ElementType::Int16: return "int16";
        case ElementType::Int32: return "int32";
        case ElementType::Int64: return "int64";
        case ElementType::Float16: return "float16";
        case ElementType::BFloat16: return "bfloat16";
        case ElementType::Float32: return "float32";
        case ElementType::Float64: return "float64";
        default: return "unknown";
    }
}

std::size_t ShapeElementCount(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    std::size_t count = 1;
    for (std::int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("Dynamic/negative Warp runtime shape is not supported");
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

std::size_t ShapeByteSize(const std::vector<std::int64_t>& shape, ElementType dtype) {
    const std::size_t item_size = ElementSize(dtype);
    if (item_size == 0) {
        return 0;
    }
    return ShapeElementCount(shape) * item_size;
}

}  // namespace leapp::warp_runtime
