#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "../core/runtime_metadata.h"
#include "../core/warp_apic_runner.h"

namespace leapp::warp_runtime::torch_adapter {
namespace {

ElementType FromTorchType(at::ScalarType dtype) {
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

at::ScalarType ToTorchType(ElementType dtype) {
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

TensorView FromTensor(const at::Tensor& tensor) {
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
    c10::weak_intrusive_ptr<c10::StorageImpl> bundle_storage;
    std::unique_ptr<WarpApicRunner> runner;
};

std::mutex g_cache_mutex;
std::unordered_map<std::string, std::shared_ptr<RunnerCacheEntry>> g_cache;

// Called with g_cache_mutex held. Model constants own their storage.
void PruneRunners() {
    for (auto it = g_cache.begin(); it != g_cache.end();) {
        if (it->second->bundle_storage.expired()) {
            it = g_cache.erase(it);
        } else {
            ++it;
        }
    }
}

std::shared_ptr<RunnerCacheEntry> GetRunner(const std::string& runtime_metadata,
                                            const at::Tensor& bundle) {
    const std::string cache_key =
        runtime_metadata + ":" +
        std::to_string(reinterpret_cast<std::uintptr_t>(bundle.data_ptr()));
    std::lock_guard<std::mutex> guard(g_cache_mutex);
    PruneRunners();
    auto it = g_cache.find(cache_key);
    if (it != g_cache.end()) {
        return it->second;
    }
    auto metadata = ParseRuntimeMetadata(runtime_metadata);
    auto runner = std::make_unique<WarpApicRunner>(metadata);
    const at::Tensor cpu_bundle = bundle.device().is_cpu() ? bundle.contiguous() : bundle.cpu().contiguous();
    runner->LoadOnce(static_cast<const std::uint8_t*>(cpu_bundle.data_ptr()), cpu_bundle.nbytes());
    auto entry = std::make_shared<RunnerCacheEntry>(RunnerCacheEntry{
        std::move(metadata), bundle.storage().getWeakStorageImpl(), std::move(runner)});
    g_cache.emplace(cache_key, entry);
    return entry;
}

std::vector<at::Tensor> WarpRunner(const std::vector<at::Tensor>& inputs,
                                   const std::string& runtime_metadata,
                                   const at::Tensor& bundle) {
    TORCH_CHECK(!bundle.is_cuda() || bundle.get_device() == 0,
                "Warp replay currently requires CUDA device 0");
    for (const auto& input : inputs) {
        TORCH_CHECK(!input.is_cuda() || input.get_device() == 0,
                    "Warp replay currently requires CUDA device 0");
    }
    const c10::cuda::CUDAGuard device_guard(0);
    auto entry = GetRunner(runtime_metadata, bundle);
    const auto& metadata = entry->metadata;
    std::vector<at::Tensor> outputs;
    outputs.reserve(metadata.outputs.size());
    const at::Device device = inputs.empty() ? bundle.device() : inputs.front().device();
    for (const auto& spec : metadata.outputs) {
        std::vector<std::int64_t> shape = spec.shape;
        outputs.push_back(at::empty(shape, at::TensorOptions().dtype(ToTorchType(spec.dtype)).device(device)));
    }

    std::vector<at::Tensor> contiguous_inputs;
    contiguous_inputs.reserve(inputs.size());
    RuntimeInvocation invocation;
    invocation.inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        contiguous_inputs.push_back(input.contiguous());
        invocation.inputs.push_back(FromTensor(contiguous_inputs.back()));
    }
    invocation.outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
        invocation.outputs.push_back(FromTensor(output));
    }
    invocation.cuda_stream = c10::cuda::getCurrentCUDAStream(0).stream();
    entry->runner->Run(invocation);
    return outputs;
}

}  // namespace
}  // namespace leapp::warp_runtime::torch_adapter

TORCH_LIBRARY_FRAGMENT(leapp, m) {
    // Python may have already defined the schema while exporting a model.
    if (!c10::Dispatcher::singleton().findSchema({"leapp::warp_runner", ""})) {
        m.def("warp_runner(Tensor[] inputs, str runtime_metadata, Tensor bundle) -> Tensor[]");
    }
}

TORCH_LIBRARY_IMPL(leapp, CPU, m) {
    m.impl("warp_runner", leapp::warp_runtime::torch_adapter::WarpRunner);
}

TORCH_LIBRARY_IMPL(leapp, CUDA, m) {
    m.impl("warp_runner", leapp::warp_runtime::torch_adapter::WarpRunner);
}

// Call after releasing model constants to reclaim unused graphs immediately.
extern "C" int LEAPP_WARP_ReleaseUnusedRunners() noexcept {
    try {
        std::lock_guard<std::mutex> guard(leapp::warp_runtime::torch_adapter::g_cache_mutex);
        leapp::warp_runtime::torch_adapter::PruneRunners();
        return 0;
    } catch (...) {
        return 1;
    }
}
