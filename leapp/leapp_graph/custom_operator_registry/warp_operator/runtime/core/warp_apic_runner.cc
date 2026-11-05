#include "warp_apic_runner.h"

#include "wrpb_archive.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "warp.h"

namespace leapp::warp_runtime {
namespace {

constexpr const char* kWrpFilename = "segment.wrp";

class ScopedCudaContext {
 public:
    explicit ScopedCudaContext(void* context) : previous_(wp_cuda_context_get_current()) {
        wp_cuda_context_set_current(context);
    }
    ~ScopedCudaContext() { wp_cuda_context_set_current(previous_); }
    ScopedCudaContext(const ScopedCudaContext&) = delete;
    ScopedCudaContext& operator=(const ScopedCudaContext&) = delete;

 private:
    void* previous_;
};

std::string WarpErrorString() {
    const char* error = wp_get_error_string();
    return error == nullptr || error[0] == '\0' ? "unknown Warp error" : error;
}

void CheckCuda(cudaError_t status, const std::string& what) {
    if (status != cudaSuccess) {
        const char* msg = cudaGetErrorString(status);
        throw std::runtime_error(what + ": CUDA error " + std::to_string(status) +
                                 (msg != nullptr ? std::string(" (") + msg + ")" : std::string()));
    }
}

void ValidateParamSize(APICGraph* graph, const std::string& name, std::size_t bytes) {
    const std::size_t expected = wp_apic_get_param_size(graph, name.c_str());
    if (expected != bytes) {
        throw std::runtime_error("APIC param '" + name + "' byte size mismatch: metadata/tensor has " +
                                 std::to_string(bytes) + " bytes, APIC expects " +
                                 std::to_string(expected));
    }
}

void* ParamData(APICGraph* graph, const std::string& name) {
    void* data = wp_apic_get_param_ptr(graph, name.c_str());
    if (data == nullptr) {
        throw std::runtime_error("Missing APIC param pointer: " + name);
    }
    return data;
}

}  // namespace

WarpApicRunner::WarpApicRunner(RuntimeMetadata metadata) : metadata_(std::move(metadata)) {}

WarpApicRunner::~WarpApicRunner() {
    if (graph_ != nullptr) {
        const ScopedCudaContext context(cuda_context_);
        // A caller can release a runner after enqueueing its last CUDA invocation.
        cudaEventSynchronize(completed_);
        wp_apic_destroy_graph(graph_);
        cudaEventDestroy(completed_);
    }
}

void WarpApicRunner::EnsureCudaInitialized() {
    static std::once_flag init_once;
    static void* cached_context = nullptr;
    std::call_once(init_once, []() {
        if (wp_init(nullptr) != 0) {
            throw std::runtime_error("Failed to initialize Warp: " + WarpErrorString());
        }
        if (!wp_is_cuda_enabled() || wp_cuda_device_get_count() == 0) {
            throw std::runtime_error("Warp CUDA runtime is not available");
        }
        cached_context = wp_cuda_device_get_primary_context(0);
        if (cached_context == nullptr) {
            throw std::runtime_error("Failed to get CUDA primary context: " + WarpErrorString());
        }
    });
    cuda_context_ = cached_context;
}

void WarpApicRunner::LoadOnce(const std::uint8_t* bundle_data, std::size_t bundle_size) {
    std::call_once(load_once_, [&]() {
        if (bundle_data == nullptr || bundle_size == 0) {
            throw std::runtime_error("Warp APIC bundle input is empty");
        }
        EnsureCudaInitialized();
        const ScopedCudaContext context(cuda_context_);
        auto directory = std::make_unique<TempBundleDir>();
        ExtractWRPBToDirectory(bundle_data, bundle_size, directory->path());
        const std::filesystem::path wrp_path = directory->path() / kWrpFilename;
        std::unique_ptr<APICGraph, decltype(&wp_apic_destroy_graph)> graph(
            wp_apic_load_graph(cuda_context_, wrp_path.string().c_str(), APIC_DEVICE_CUDA),
            wp_apic_destroy_graph);
        if (graph == nullptr) {
            throw std::runtime_error("Failed to load embedded APIC graph '" + std::string(kWrpFilename) +
                                     "': " + WarpErrorString());
        }
        for (const auto& spec : metadata_.inputs) {
            ValidateParamSize(graph.get(), spec.param_name, spec.num_bytes);
        }
        for (const auto& spec : metadata_.outputs) {
            if (spec.mask) {
                ValidateParamSize(graph.get(), spec.param_name, spec.num_bytes);
            }
        }
        CheckCuda(cudaEventCreateWithFlags(&completed_, cudaEventDisableTiming),
                  "create Warp completion event");
        temp_dir_ = std::move(directory);
        graph_ = graph.release();
    });
}

void WarpApicRunner::ValidateTensor(const TensorView& tensor) const {
    if (tensor.num_bytes > 0 && tensor.data == nullptr) {
        throw std::runtime_error("Warp tensor has no storage");
    }
    if (tensor.is_cuda && tensor.device_index != 0) {
        throw std::runtime_error("Warp replay currently requires CUDA device 0");
    }
}

void WarpApicRunner::ValidateInput(const TensorView& tensor, const InputSpec& spec) const {
    ValidateTensor(tensor);
    if (tensor.dtype != spec.dtype) {
        throw std::runtime_error("Warp input '" + spec.param_name + "' dtype mismatch: got " +
                                 ElementTypeName(tensor.dtype) + ", expected " + ElementTypeName(spec.dtype));
    }
    if (tensor.shape != spec.shape || tensor.num_bytes != spec.num_bytes) {
        throw std::runtime_error("Warp input '" + spec.param_name + "' shape/byte size mismatch");
    }
}

void WarpApicRunner::ValidateOutput(const TensorView& tensor, const OutputSpec& spec) const {
    ValidateTensor(tensor);
    if (tensor.dtype != spec.dtype) {
        throw std::runtime_error("Warp output dtype mismatch at logical index " + std::to_string(spec.logical_index));
    }
    if (tensor.shape != spec.shape || tensor.num_bytes != spec.num_bytes) {
        throw std::runtime_error("Warp output shape/byte size mismatch at logical index " +
                                 std::to_string(spec.logical_index));
    }
}

void WarpApicRunner::Run(const RuntimeInvocation& invocation) {
    std::lock_guard<std::mutex> guard(run_mutex_);
    if (graph_ == nullptr) {
        throw std::runtime_error("Warp APIC graph has not been loaded");
    }
    if (invocation.inputs.size() != metadata_.inputs.size()) {
        throw std::runtime_error("Warp input count mismatch");
    }
    if (invocation.outputs.size() != metadata_.outputs.size()) {
        throw std::runtime_error("Warp output count mismatch");
    }

    bool uses_host_storage = false;
    for (std::size_t i = 0; i < metadata_.inputs.size(); ++i) {
        ValidateInput(invocation.inputs[i], metadata_.inputs[i]);
        uses_host_storage |= !invocation.inputs[i].is_cuda;
    }
    for (std::size_t i = 0; i < metadata_.outputs.size(); ++i) {
        ValidateOutput(invocation.outputs[i], metadata_.outputs[i]);
        uses_host_storage |= !invocation.outputs[i].is_cuda;
    }

    const ScopedCudaContext context(cuda_context_);
    const auto stream = static_cast<cudaStream_t>(invocation.cuda_stream);
    try {
        // Protect graph-owned buffers even when consecutive calls use new streams.
        CheckCuda(cudaStreamWaitEvent(stream, completed_, 0), "wait for Warp replay");
        for (std::size_t i = 0; i < metadata_.inputs.size(); ++i) {
            const auto& spec = metadata_.inputs[i];
            const TensorView& tensor = invocation.inputs[i];
            if (spec.num_bytes != 0) {
                CheckCuda(cudaMemcpyAsync(ParamData(graph_, spec.param_name), tensor.data, spec.num_bytes,
                                          tensor.is_cuda ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                                          stream),
                          "copy into APIC param '" + spec.param_name + "'");
            }
        }

        if (!wp_apic_launch(graph_, stream)) {
            throw std::runtime_error("Failed to launch APIC graph: " + WarpErrorString());
        }

        for (std::size_t i = 0; i < metadata_.outputs.size(); ++i) {
            const auto& spec = metadata_.outputs[i];
            const TensorView& tensor = invocation.outputs[i];
            if (spec.num_bytes == 0) {
                continue;
            }
            if (!spec.mask) {
                if (tensor.is_cuda) {
                    CheckCuda(cudaMemsetAsync(tensor.data, 0, spec.num_bytes, stream),
                              "fill unused Warp output");
                } else {
                    std::memset(tensor.data, 0, spec.num_bytes);
                }
            } else {
                CheckCuda(cudaMemcpyAsync(tensor.data, ParamData(graph_, spec.param_name), spec.num_bytes,
                                          tensor.is_cuda ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost,
                                          stream),
                          "copy from APIC param '" + spec.param_name + "'");
            }
        }
        CheckCuda(cudaEventRecord(completed_, stream), "record Warp completion");
        if (uses_host_storage) {
            CheckCuda(cudaStreamSynchronize(stream), "finish Warp host copies");
        }
    } catch (...) {
        // Finish partial submissions before callers release their tensor storage.
        cudaStreamSynchronize(stream);
        throw;
    }
}

}  // namespace leapp::warp_runtime
