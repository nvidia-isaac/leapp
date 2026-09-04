#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"

#include "../core/runtime_metadata.h"
#include "../core/warp_apic_runner.h"

namespace leapp::warp_runtime::onnx_adapter {
namespace {

const OrtApi* g_ort = nullptr;
constexpr const char* kDomain = "com.nvidia.warp";
constexpr const char* kOpName = "WrpRunner";

void CheckOrt(OrtStatus* status) {
    if (status == nullptr) {
        return;
    }
    std::string message = g_ort->GetErrorMessage(status);
    g_ort->ReleaseStatus(status);
    throw std::runtime_error(message);
}

std::string GetStringAttribute(const OrtKernelInfo* info, const char* name) {
    std::size_t size = 0;
    OrtStatus* status = g_ort->KernelInfoGetAttribute_string(info, name, nullptr, &size);
    if (status != nullptr) {
        g_ort->ReleaseStatus(status);
    }
    if (size == 0) {
        throw std::runtime_error(std::string("Missing string attribute: ") + name);
    }
    std::string value(size, '\0');
    CheckOrt(g_ort->KernelInfoGetAttribute_string(info, name, value.data(), &size));
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

TensorView TensorViewFromOrt(const OrtValue* value, bool is_cuda) {
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    CheckOrt(g_ort->GetTensorTypeAndShape(value, &shape_info));
    ONNXTensorElementDataType ort_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    CheckOrt(g_ort->GetTensorElementType(shape_info, &ort_dtype));
    std::size_t rank = 0;
    CheckOrt(g_ort->GetDimensionsCount(shape_info, &rank));
    std::vector<std::int64_t> dims(rank);
    CheckOrt(g_ort->GetDimensions(shape_info, dims.data(), rank));
    std::size_t elem_count = 0;
    CheckOrt(g_ort->GetTensorShapeElementCount(shape_info, &elem_count));
    g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    void* data = nullptr;
    CheckOrt(g_ort->GetTensorMutableData(const_cast<OrtValue*>(value), &data));
    TensorView view;
    view.data = data;
    view.dtype = FromOrtType(ort_dtype);
    view.shape = std::move(dims);
    view.num_bytes = elem_count * ElementSize(view.dtype);
    view.is_cuda = is_cuda;
    return view;
}

struct Kernel {
    explicit Kernel(const OrtKernelInfo* info)
        : metadata(ParseRuntimeMetadata(GetStringAttribute(info, "runtime_metadata"))),
          runner(std::make_unique<WarpApicRunner>(metadata)) {}

    void Compute(OrtKernelContext* ctx) {
        std::size_t input_count = 0;
        std::size_t output_count = 0;
        CheckOrt(g_ort->KernelContext_GetInputCount(ctx, &input_count));
        CheckOrt(g_ort->KernelContext_GetOutputCount(ctx, &output_count));
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
        CheckOrt(g_ort->KernelContext_GetInput(ctx, bundle_index, &bundle));
        TensorView bundle_view = TensorViewFromOrt(bundle, false);
        if (bundle_view.dtype != ElementType::UInt8) {
            throw std::runtime_error("WrpRunner bundle input must be tensor(uint8)");
        }
        runner->LoadOnce(static_cast<const std::uint8_t*>(bundle_view.data), bundle_view.num_bytes);

        RuntimeInvocation invocation;
        invocation.inputs.reserve(metadata.inputs.size());
        for (std::size_t i = 0; i < metadata.inputs.size(); ++i) {
            const OrtValue* input = nullptr;
            CheckOrt(g_ort->KernelContext_GetInput(ctx, i, &input));
            invocation.inputs.push_back(TensorViewFromOrt(input, false));
        }

        for (std::size_t i = 0; i < metadata.outputs.size(); ++i) {
            const auto& spec = metadata.outputs[i];
            OrtValue* output = nullptr;
            CheckOrt(g_ort->KernelContext_GetOutput(
                ctx,
                i,
                spec.shape.data(),
                spec.shape.size(),
                &output));
            invocation.outputs.push_back(TensorViewFromOrt(output, false));
        }

        invocation.cuda_stream = nullptr;
        runner->Run(invocation);
    }

    RuntimeMetadata metadata;
    std::unique_ptr<WarpApicRunner> runner;
};

OrtStatusPtr CreateKernelV2(const OrtCustomOp*, const OrtApi*, const OrtKernelInfo* info, void** kernel) {
    try {
        *kernel = new Kernel(info);
        return nullptr;
    } catch (const std::exception& e) {
        return g_ort->CreateStatus(ORT_FAIL, e.what());
    }
}

void* CreateKernel(const OrtCustomOp*, const OrtApi*, const OrtKernelInfo* info) {
    try {
        return new Kernel(info);
    } catch (...) {
        return nullptr;
    }
}

void DestroyKernel(void* kernel) { delete static_cast<Kernel*>(kernel); }
void KernelCompute(void* kernel, OrtKernelContext* context) { static_cast<Kernel*>(kernel)->Compute(context); }

OrtStatusPtr KernelComputeV2(void* kernel, OrtKernelContext* context) {
    try {
        static_cast<Kernel*>(kernel)->Compute(context);
        return nullptr;
    } catch (const std::exception& e) {
        return g_ort->CreateStatus(ORT_FAIL, e.what());
    }
}

const char* GetName(const OrtCustomOp*) { return kOpName; }
const char* GetExecutionProviderType(const OrtCustomOp*) { return nullptr; }

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

OrtCustomOp CreateOp() {
    OrtCustomOp op{};
    op.version = ORT_API_VERSION;
    op.CreateKernel = CreateKernel;
    op.GetName = GetName;
    op.GetExecutionProviderType = GetExecutionProviderType;
    op.GetInputTypeCount = GetInputTypeCount;
    op.GetInputType = GetInputType;
    op.GetOutputTypeCount = GetOutputTypeCount;
    op.GetOutputType = GetOutputType;
    op.KernelCompute = KernelCompute;
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

OrtCustomOp g_op = CreateOp();

}  // namespace
}  // namespace leapp::warp_runtime::onnx_adapter

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                      const OrtApiBase* api_base) {
    using namespace leapp::warp_runtime::onnx_adapter;
    g_ort = api_base->GetApi(ORT_API_VERSION);
    OrtCustomOpDomain* domain = nullptr;
    OrtStatus* status = g_ort->CreateCustomOpDomain(kDomain, &domain);
    if (status != nullptr) {
        return status;
    }
    status = g_ort->CustomOpDomain_Add(domain, &g_op);
    if (status != nullptr) {
        return status;
    }
    return g_ort->AddCustomOpDomain(options, domain);
}
