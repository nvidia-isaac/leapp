#include <memory>
#include <mutex>
#include <unordered_map>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "../core/runtime_metadata.h"
#include "../core/warp_apic_runner.h"

namespace leapp::warp_runtime::torch_adapter {
namespace {

ElementType FromTorchType(c10::ScalarType dtype) {
    switch (dtype) {
        case c10::kBool: return ElementType::Bool;
        case c10::kByte: return ElementType::UInt8;
        case c10::kChar: return ElementType::Int8;
        case c10::kShort: return ElementType::Int16;
        case c10::kInt: return ElementType::Int32;
        case c10::kLong: return ElementType::Int64;
        case c10::kHalf: return ElementType::Float16;
        case c10::kBFloat16: return ElementType::BFloat16;
        case c10::kFloat: return ElementType::Float32;
        case c10::kDouble: return ElementType::Float64;
        default: return ElementType::Unknown;
    }
}

torch::ScalarType ToTorchType(ElementType dtype) {
    switch (dtype) {
        case ElementType::Bool: return c10::kBool;
        case ElementType::UInt8: return c10::kByte;
        case ElementType::Int8: return c10::kChar;
        case ElementType::Int16: return c10::kShort;
        case ElementType::Int32: return c10::kInt;
        case ElementType::Int64: return c10::kLong;
        case ElementType::Float16: return c10::kHalf;
        case ElementType::BFloat16: return c10::kBFloat16;
        case ElementType::Float32: return c10::kFloat;
        case ElementType::Float64: return c10::kDouble;
        default: throw std::runtime_error("Unsupported Warp output dtype");
    }
}

TensorView FromTensor(const torch::Tensor& tensor) {
    TensorView view;
    view.data = tensor.data_ptr();
    view.dtype = FromTorchType(tensor.scalar_type());
    view.shape.assign(tensor.sizes().begin(), tensor.sizes().end());
    view.num_bytes = tensor.nbytes();
    view.is_cuda = tensor.is_cuda();
    view.device_index = tensor.device().has_index() ? tensor.device().index() : 0;
    return view;
}

struct RunnerCacheEntry {
    RuntimeMetadata metadata;
    std::unique_ptr<WarpApicRunner> runner;
};

std::mutex g_cache_mutex;
std::unordered_map<std::string, std::shared_ptr<RunnerCacheEntry>> g_cache;

std::shared_ptr<RunnerCacheEntry> GetRunner(const std::string& runtime_metadata,
                                            const torch::Tensor& bundle) {
    std::lock_guard<std::mutex> guard(g_cache_mutex);
    auto it = g_cache.find(runtime_metadata);
    if (it != g_cache.end()) {
        return it->second;
    }
    auto entry = std::make_shared<RunnerCacheEntry>();
    entry->metadata = ParseRuntimeMetadata(runtime_metadata);
    entry->runner = std::make_unique<WarpApicRunner>(entry->metadata);
    const torch::Tensor cpu_bundle = bundle.device().is_cpu() ? bundle.contiguous() : bundle.cpu().contiguous();
    entry->runner->LoadOnce(static_cast<const std::uint8_t*>(cpu_bundle.data_ptr()), cpu_bundle.nbytes());
    g_cache.emplace(runtime_metadata, entry);
    return entry;
}

std::vector<torch::Tensor> WarpRunner(const std::vector<torch::Tensor>& inputs,
                                      const std::string& runtime_metadata,
                                      const torch::Tensor& bundle) {
    auto entry = GetRunner(runtime_metadata, bundle);
    const auto& metadata = entry->metadata;
    std::vector<torch::Tensor> outputs;
    outputs.reserve(metadata.outputs.size());
    const torch::Device device = inputs.empty() ? torch::Device(torch::kCUDA, 0) : inputs.front().device();
    for (const auto& spec : metadata.outputs) {
        std::vector<std::int64_t> shape = spec.shape;
        outputs.push_back(torch::empty(shape, torch::TensorOptions().dtype(ToTorchType(spec.dtype)).device(device)));
    }

    RuntimeInvocation invocation;
    invocation.inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        invocation.inputs.push_back(FromTensor(input));
    }
    invocation.outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
        invocation.outputs.push_back(FromTensor(output));
    }
    invocation.cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    entry->runner->Run(invocation);
    return outputs;
}

}  // namespace
}  // namespace leapp::warp_runtime::torch_adapter

TORCH_LIBRARY_FRAGMENT(leapp, m) {
    m.def("warp_runner(Tensor[] inputs, str runtime_metadata, Tensor bundle) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(leapp, CUDA, m) {
    m.impl("warp_runner", leapp::warp_runtime::torch_adapter::WarpRunner);
}
