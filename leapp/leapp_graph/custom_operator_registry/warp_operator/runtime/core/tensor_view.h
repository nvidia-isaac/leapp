#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace leapp::warp_runtime {

enum class ElementType {
    Unknown,
    Bool,
    UInt8,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    BFloat16,
    Float32,
    Float64,
};

struct TensorView {
    void* data = nullptr;
    ElementType dtype = ElementType::Unknown;
    std::vector<std::int64_t> shape;
    std::size_t num_bytes = 0;
    bool is_cuda = false;
    int device_index = 0;
};

struct RuntimeInvocation {
    std::vector<TensorView> inputs;
    std::vector<TensorView> outputs;
    void* cuda_stream = nullptr;
};

std::size_t ElementSize(ElementType dtype);
ElementType ElementTypeFromName(const std::string& name);
std::string ElementTypeName(ElementType dtype);
std::size_t ShapeElementCount(const std::vector<std::int64_t>& shape);
std::size_t ShapeByteSize(const std::vector<std::int64_t>& shape, ElementType dtype);

}  // namespace leapp::warp_runtime
