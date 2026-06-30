#include "warp_apic_runner.h"

#include "wrpb_archive.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "warp.h"

extern "C" int cudaMemcpyAsync(void* dst, const void* src, std::size_t count, int kind, void* stream);
extern "C" int cudaMemsetAsync(void* dst, int value, std::size_t count, void* stream);
extern "C" const char* cudaGetErrorString(int error);

namespace leapp::warp_runtime {
namespace {

constexpr int kCudaSuccess = 0;
constexpr int kCudaMemcpyDeviceToDevice = 3;

std::string WarpErrorString() {
    const char* error = wp_get_error_string();
    return error == nullptr || error[0] == '\0' ? "unknown Warp error" : error;
}

void CheckCuda(int status, const std::string& what) {
    if (status != kCudaSuccess) {
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

}  // namespace

WarpApicRunner::WarpApicRunner(RuntimeMetadata metadata) : metadata_(std::move(metadata)) {}

WarpApicRunner::~WarpApicRunner() {
    if (graph_ != nullptr) {
        wp_apic_destroy_graph(graph_);
        graph_ = nullptr;
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
        temp_dir_ = std::make_unique<TempBundleDir>();
        ExtractWRPBToDirectory(bundle_data, bundle_size, temp_dir_->path());
        const std::filesystem::path wrp_path = temp_dir_->path() / metadata_.wrp_name;
        wp_cuda_context_set_current(cuda_context_);
        graph_ = wp_apic_load_graph(cuda_context_, wrp_path.string().c_str(), APIC_DEVICE_CUDA);
        if (graph_ == nullptr) {
            throw std::runtime_error("Failed to load embedded APIC graph '" + metadata_.wrp_name +
                                     "': " + WarpErrorString());
        }
        for (const auto& spec : metadata_.inputs) {
            ValidateParamSize(graph_, spec.param_name, spec.num_bytes);
        }
        for (const auto& spec : metadata_.outputs) {
            if (spec.mask) {
                ValidateParamSize(graph_, spec.param_name, spec.num_bytes);
            }
        }
    });
}

void WarpApicRunner::ValidateInput(const TensorView& tensor, const InputSpec& spec) const {
    if (tensor.dtype != spec.dtype) {
        throw std::runtime_error("Warp input '" + spec.param_name + "' dtype mismatch: got " +
                                 ElementTypeName(tensor.dtype) + ", expected " + ElementTypeName(spec.dtype));
    }
    if (tensor.shape != spec.shape || tensor.num_bytes != spec.num_bytes) {
        throw std::runtime_error("Warp input '" + spec.param_name + "' shape/byte size mismatch");
    }
}

void WarpApicRunner::ValidateOutput(const TensorView& tensor, const OutputSpec& spec) const {
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

    bool device_resident = invocation.cuda_stream != nullptr;
    for (const auto& tensor : invocation.inputs) {
        device_resident = device_resident && tensor.is_cuda;
    }
    for (const auto& tensor : invocation.outputs) {
        device_resident = device_resident && tensor.is_cuda;
    }

    wp_cuda_context_set_current(cuda_context_);
    for (std::size_t i = 0; i < metadata_.inputs.size(); ++i) {
        const auto& spec = metadata_.inputs[i];
        const TensorView& tensor = invocation.inputs[i];
        ValidateInput(tensor, spec);
        if (device_resident) {
            void* param_ptr = wp_apic_get_param_ptr(graph_, spec.param_name.c_str());
            if (param_ptr == nullptr) {
                throw std::runtime_error("Failed to get APIC input param pointer '" + spec.param_name + "'");
            }
            CheckCuda(cudaMemcpyAsync(param_ptr, tensor.data, spec.num_bytes,
                                      kCudaMemcpyDeviceToDevice, invocation.cuda_stream),
                      "copy into APIC param '" + spec.param_name + "'");
        } else {
            if (!wp_apic_set_param(graph_, spec.param_name.c_str(), tensor.data, spec.num_bytes)) {
                throw std::runtime_error("Failed to set APIC param '" + spec.param_name + "': " + WarpErrorString());
            }
        }
    }

    if (!wp_apic_launch(graph_, device_resident ? invocation.cuda_stream : nullptr)) {
        throw std::runtime_error("Failed to launch APIC graph: " + WarpErrorString());
    }
    if (!device_resident) {
        wp_cuda_context_synchronize(cuda_context_);
    }

    for (std::size_t i = 0; i < metadata_.outputs.size(); ++i) {
        const auto& spec = metadata_.outputs[i];
        const TensorView& tensor = invocation.outputs[i];
        ValidateOutput(tensor, spec);
        if (!spec.mask || spec.num_bytes == 0) {
            if (tensor.data != nullptr && tensor.num_bytes > 0) {
                if (device_resident) {
                    CheckCuda(cudaMemsetAsync(tensor.data, 0, tensor.num_bytes, invocation.cuda_stream),
                              "fill unused Warp output");
                } else {
                    std::memset(tensor.data, 0, tensor.num_bytes);
                }
            }
            continue;
        }
        if (device_resident) {
            void* param_ptr = wp_apic_get_param_ptr(graph_, spec.param_name.c_str());
            if (param_ptr == nullptr) {
                throw std::runtime_error("Failed to get APIC output param pointer '" + spec.param_name + "'");
            }
            CheckCuda(cudaMemcpyAsync(tensor.data, param_ptr, spec.num_bytes,
                                      kCudaMemcpyDeviceToDevice, invocation.cuda_stream),
                      "copy from APIC param '" + spec.param_name + "'");
        } else {
            if (!wp_apic_get_param(graph_, spec.param_name.c_str(), tensor.data, spec.num_bytes)) {
                throw std::runtime_error("Failed to get APIC param '" + spec.param_name + "': " + WarpErrorString());
            }
        }
    }
}

}  // namespace leapp::warp_runtime
