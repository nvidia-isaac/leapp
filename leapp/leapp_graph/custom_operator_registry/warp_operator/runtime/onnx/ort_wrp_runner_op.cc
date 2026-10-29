#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "onnxruntime_c_api.h"

#include "../core/runtime_metadata.h"
#include "../core/warp_apic_runner.h"

namespace leapp::warp_runtime::onnx_adapter {
namespace {

constexpr const char* kDomain = "com.nvidia.warp";
constexpr const char* kOpName = "WrpRunner";

void CheckOrt(const OrtApi& api, OrtStatus* status) {
    if (status == nullptr) {
        return;
    }
    const std::unique_ptr<OrtStatus, decltype(api.ReleaseStatus)> owner(status, api.ReleaseStatus);
    throw std::runtime_error(api.GetErrorMessage(status));
}

void CheckCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

std::string GetStringAttribute(const OrtApi& api, const OrtKernelInfo* info, const char* name) {
    std::size_t size = 0;
    OrtStatus* status = api.KernelInfoGetAttribute_string(info, name, nullptr, &size);
    if (status != nullptr) {
        api.ReleaseStatus(status);
    }
    if (size == 0) {
        throw std::runtime_error(std::string("Missing string attribute: ") + name);
    }
    std::string value(size, '\0');
    CheckOrt(api, api.KernelInfoGetAttribute_string(info, name, value.data(), &size));
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

ElementType FromOrtType(ONNXTensorElementDataType dtype) {
    switch (dtype) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return ElementType::Bool;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return ElementType::UInt8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return ElementType::Int8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return ElementType::Int16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return ElementType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return ElementType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return ElementType::Float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return ElementType::BFloat16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return ElementType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return ElementType::Float64;
        default: return ElementType::Unknown;
    }
}

ONNXTensorElementDataType ToOrtType(ElementType dtype) {
    switch (dtype) {
        case ElementType::Bool: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        case ElementType::UInt8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case ElementType::Int8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case ElementType::Int16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case ElementType::Int32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case ElementType::Int64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case ElementType::Float16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case ElementType::BFloat16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        case ElementType::Float32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case ElementType::Float64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

TensorView TensorViewFromOrt(const OrtApi& api, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    CheckOrt(api, api.GetTensorTypeAndShape(value, &shape_info));
    const std::unique_ptr<OrtTensorTypeAndShapeInfo, decltype(api.ReleaseTensorTypeAndShapeInfo)>
        owner(shape_info, api.ReleaseTensorTypeAndShapeInfo);
    ONNXTensorElementDataType ort_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    CheckOrt(api, api.GetTensorElementType(shape_info, &ort_dtype));
    std::size_t rank = 0;
    CheckOrt(api, api.GetDimensionsCount(shape_info, &rank));
    std::vector<std::int64_t> dims(rank);
    CheckOrt(api, api.GetDimensions(shape_info, dims.data(), rank));
    std::size_t elem_count = 0;
    CheckOrt(api, api.GetTensorShapeElementCount(shape_info, &elem_count));

    void* data = nullptr;
    CheckOrt(api, api.GetTensorMutableData(const_cast<OrtValue*>(value), &data));
    TensorView view;
    view.data = data;
    view.dtype = FromOrtType(ort_dtype);
    view.shape = std::move(dims);
    view.num_bytes = elem_count * ElementSize(view.dtype);
    const OrtMemoryInfo* memory_info = nullptr;
    CheckOrt(api, api.GetTensorMemoryInfo(value, &memory_info));
    OrtMemoryInfoDeviceType device_type;
    api.MemoryInfoGetDeviceType(memory_info, &device_type);
    view.is_cuda = device_type == OrtMemoryInfoDeviceType_GPU;
    CheckOrt(api, api.MemoryInfoGetId(memory_info, &view.device_index));
    if (view.is_cuda && view.device_index != 0) {
        throw std::runtime_error("Warp replay currently requires CUDA device 0");
    }
    return view;
}

struct Kernel {
    Kernel(const OrtApi& api, const OrtKernelInfo* info)
        : api(api), metadata(ParseRuntimeMetadata(GetStringAttribute(api, info, "runtime_metadata"))),
          runner(std::make_unique<WarpApicRunner>(metadata)) {}

    void Compute(OrtKernelContext* ctx) {
        std::size_t input_count = 0;
        std::size_t output_count = 0;
        CheckOrt(api, api.KernelContext_GetInputCount(ctx, &input_count));
        CheckOrt(api, api.KernelContext_GetOutputCount(ctx, &output_count));
        if (input_count == 0) {
            throw std::runtime_error("WrpRunner expected at least one bundle input");
        }
        const std::size_t bundle_index = input_count - 1;
        if (bundle_index != metadata.inputs.size()) {
            throw std::runtime_error("WrpRunner data input count does not match runtime_metadata");
        }
        if (output_count != metadata.outputs.size()) {
            throw std::runtime_error("WrpRunner output count does not match runtime_metadata");
        }

        const OrtValue* bundle = nullptr;
        CheckOrt(api, api.KernelContext_GetInput(ctx, bundle_index, &bundle));
        RuntimeInvocation invocation;
        CheckOrt(api, api.KernelContext_GetGPUComputeStream(ctx, &invocation.cuda_stream));
        TensorView bundle_view = TensorViewFromOrt(api, bundle);
        if (bundle_view.dtype != ElementType::UInt8) {
            throw std::runtime_error("WrpRunner bundle input must be tensor(uint8)");
        }
        std::call_once(load_once, [&]() {
            // Variadic CUDA inputs include the archive. Only loading needs it on host.
            std::vector<std::uint8_t> host_bundle;
            const auto* data = static_cast<const std::uint8_t*>(bundle_view.data);
            if (bundle_view.is_cuda) {
                host_bundle.resize(bundle_view.num_bytes);
                auto stream = static_cast<cudaStream_t>(invocation.cuda_stream);
                CheckCuda(cudaMemcpyAsync(host_bundle.data(), data, host_bundle.size(),
                                          cudaMemcpyDeviceToHost, stream));
                CheckCuda(cudaStreamSynchronize(stream));
                data = host_bundle.data();
            }
            runner->LoadOnce(data, bundle_view.num_bytes);
        });
        invocation.inputs.reserve(metadata.inputs.size());
        for (std::size_t i = 0; i < metadata.inputs.size(); ++i) {
            const OrtValue* input = nullptr;
            CheckOrt(api, api.KernelContext_GetInput(ctx, i, &input));
            invocation.inputs.push_back(TensorViewFromOrt(api, input));
        }

        for (std::size_t i = 0; i < metadata.outputs.size(); ++i) {
            const auto& spec = metadata.outputs[i];
            OrtValue* output = nullptr;
            CheckOrt(api, api.KernelContext_GetOutput(
                ctx,
                i,
                spec.shape.data(),
                spec.shape.size(),
                &output));
            invocation.outputs.push_back(TensorViewFromOrt(api, output));
        }

        runner->Run(invocation);
    }

    const OrtApi& api;
    std::once_flag load_once;
    RuntimeMetadata metadata;
    std::unique_ptr<WarpApicRunner> runner;
};

OrtStatusPtr CreateKernelV2(const OrtCustomOp*, const OrtApi* api, const OrtKernelInfo* info, void** kernel) {
    try {
        *kernel = new Kernel(*api, info);
        return nullptr;
    } catch (const std::exception& e) {
        return api->CreateStatus(ORT_FAIL, e.what());
    } catch (...) {
        return api->CreateStatus(ORT_FAIL, "Unknown exception creating Warp kernel");
    }
}

void DestroyKernel(void* kernel) { delete static_cast<Kernel*>(kernel); }

OrtStatusPtr KernelComputeV2(void* kernel, OrtKernelContext* context) {
    auto* value = static_cast<Kernel*>(kernel);
    try {
        value->Compute(context);
        return nullptr;
    } catch (const std::exception& e) {
        return value->api.CreateStatus(ORT_FAIL, e.what());
    } catch (...) {
        return value->api.CreateStatus(ORT_FAIL, "Unknown exception in Warp kernel");
    }
}

struct WarpOp : OrtCustomOp {
    const char* provider;
};

const char* GetName(const OrtCustomOp*) { return kOpName; }
const char* GetExecutionProviderType(const OrtCustomOp* op) {
    return static_cast<const WarpOp*>(op)->provider;
}

std::size_t GetInputTypeCount(const OrtCustomOp*) { return 1; }
ONNXTensorElementDataType GetInputType(const OrtCustomOp*, std::size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}
std::size_t GetOutputTypeCount(const OrtCustomOp*) { return 1; }
ONNXTensorElementDataType GetOutputType(const OrtCustomOp*, std::size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}
OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(const OrtCustomOp*, std::size_t) {
    return INPUT_OUTPUT_VARIADIC;
}
OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(const OrtCustomOp*, std::size_t) {
    return INPUT_OUTPUT_VARIADIC;
}
OrtMemType GetInputMemoryType(const OrtCustomOp*, std::size_t) { return OrtMemTypeDefault; }
int GetVariadicInputMinArity(const OrtCustomOp*) { return 1; }
int GetVariadicInputHomogeneity(const OrtCustomOp*) { return 0; }
int GetVariadicOutputMinArity(const OrtCustomOp*) { return 1; }
int GetVariadicOutputHomogeneity(const OrtCustomOp*) { return 0; }
int GetStartVersion(const OrtCustomOp*) { return 1; }
int GetEndVersion(const OrtCustomOp*) { return 1; }

WarpOp CreateOp(const char* provider) {
    WarpOp op{};
    op.provider = provider;
    op.version = ORT_API_VERSION;
    op.GetName = GetName;
    op.GetExecutionProviderType = GetExecutionProviderType;
    op.GetInputTypeCount = GetInputTypeCount;
    op.GetInputType = GetInputType;
    op.GetOutputTypeCount = GetOutputTypeCount;
    op.GetOutputType = GetOutputType;
    op.KernelDestroy = DestroyKernel;
    op.GetInputCharacteristic = GetInputCharacteristic;
    op.GetOutputCharacteristic = GetOutputCharacteristic;
    op.GetInputMemoryType = GetInputMemoryType;
    op.GetVariadicInputMinArity = GetVariadicInputMinArity;
    op.GetVariadicInputHomogeneity = GetVariadicInputHomogeneity;
    op.GetVariadicOutputMinArity = GetVariadicOutputMinArity;
    op.GetVariadicOutputHomogeneity = GetVariadicOutputHomogeneity;
    op.CreateKernelV2 = CreateKernelV2;
    op.KernelComputeV2 = KernelComputeV2;
    op.GetStartVersion = GetStartVersion;
    op.GetEndVersion = GetEndVersion;
    return op;
}

OrtCustomOpDomain* GetDomain(const OrtApi& api) {
    static const WarpOp cpu_op = CreateOp(nullptr);
    static const WarpOp cuda_op = CreateOp("CUDAExecutionProvider");
    using Domain = std::unique_ptr<OrtCustomOpDomain, decltype(api.ReleaseCustomOpDomain)>;
    // Domains must outlive every session using them; reuse one per runtime API.
    static std::vector<std::pair<const OrtApi*, Domain>> domains;
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    for (const auto& entry : domains) {
        if (entry.first == &api) {
            return entry.second.get();
        }
    }
    OrtCustomOpDomain* value = nullptr;
    CheckOrt(api, api.CreateCustomOpDomain(kDomain, &value));
    Domain domain(value, api.ReleaseCustomOpDomain);
    CheckOrt(api, api.CustomOpDomain_Add(value, &cpu_op));
    CheckOrt(api, api.CustomOpDomain_Add(value, &cuda_op));
    domains.emplace_back(&api, std::move(domain));
    return value;
}

}  // namespace
}  // namespace leapp::warp_runtime::onnx_adapter

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                      const OrtApiBase* api_base) {
    using namespace leapp::warp_runtime::onnx_adapter;
    const auto* api = api_base->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
        return api_base->GetApi(1)->CreateStatus(
            ORT_FAIL, "Warp adapter was built for a newer ONNX Runtime API");
    }
    try {
        return api->AddCustomOpDomain(options, GetDomain(*api));
    } catch (const std::exception& e) {
        return api->CreateStatus(ORT_FAIL, e.what());
    } catch (...) {
        return api->CreateStatus(ORT_FAIL, "Unknown exception registering Warp ops");
    }
}
