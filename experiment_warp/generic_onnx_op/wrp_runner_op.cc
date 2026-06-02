#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "apic.h"
#include "onnxruntime_c_api.h"
#include "warp.h"

namespace {

const OrtApi* g_ort = nullptr;

constexpr const char* kDomain = "com.nvidia.warp";
constexpr const char* kOpName = "WrpRunner";
constexpr int kInputCount = 2;
constexpr int kOutputCount = 1;

void CheckOrt(OrtStatus* status) {
    if (status == nullptr) {
        return;
    }
    std::string message = g_ort->GetErrorMessage(status);
    g_ort->ReleaseStatus(status);
    throw std::runtime_error(message);
}

std::string WarpErrorString() {
    const char* error = wp_get_error_string();
    if (error == nullptr || error[0] == '\0') {
        return "unknown Warp error";
    }
    return error;
}

std::vector<std::string> SplitCsv(const std::string& value) {
    std::vector<std::string> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const auto first = item.find_first_not_of(" \t\n\r");
        const auto last = item.find_last_not_of(" \t\n\r");
        if (first == std::string::npos) {
            continue;
        }
        result.push_back(item.substr(first, last - first + 1));
    }
    return result;
}

std::vector<int64_t> ParseShapeAttribute(const std::string& value) {
    std::vector<int64_t> shape;
    for (const std::string& dim : SplitCsv(value)) {
        try {
            int64_t parsed = std::stoll(dim);
            if (parsed <= 0) {
                throw std::runtime_error("dimension must be positive");
            }
            shape.push_back(parsed);
        } catch (const std::exception& e) {
            throw std::runtime_error("Invalid output_shape dimension '" + dim + "': " + e.what());
        }
    }
    if (shape.empty()) {
        throw std::runtime_error("output_shape must contain at least one dimension");
    }
    return shape;
}

size_t ShapeElementCount(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    return count;
}

std::string GetStringAttribute(const OrtKernelInfo* info, const char* name) {
    size_t size = 0;
    OrtStatus* status = g_ort->KernelInfoGetAttribute_string(info, name, nullptr, &size);
    if (status != nullptr) {
        g_ort->ReleaseStatus(status);
    }
    if (size == 0) {
        throw std::runtime_error(std::string("Missing or empty string attribute: ") + name);
    }

    std::string value(size, '\0');
    CheckOrt(g_ort->KernelInfoGetAttribute_string(info, name, value.data(), &size));
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

void EnsureWarpCudaInitialized(void** context) {
    static std::once_flag once;
    static void* cached_context = nullptr;

    std::call_once(once, []() {
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

    *context = cached_context;
}

struct WrpRunnerKernel {
    explicit WrpRunnerKernel(const OrtKernelInfo* info) {
        wrp_path = GetStringAttribute(info, "wrp_path");
        input_names = SplitCsv(GetStringAttribute(info, "input_names"));
        output_names = SplitCsv(GetStringAttribute(info, "output_names"));
        output_shape = ParseShapeAttribute(GetStringAttribute(info, "output_shape"));
        output_element_count = ShapeElementCount(output_shape);

        if (input_names.size() != kInputCount) {
            throw std::runtime_error("WrpRunner prototype expects exactly 2 input_names");
        }
        if (output_names.size() != kOutputCount) {
            throw std::runtime_error("WrpRunner prototype expects exactly 1 output_name");
        }

        EnsureWarpCudaInitialized(&context);
        wp_cuda_context_set_current(context);
        graph = wp_apic_load_graph(context, wrp_path.c_str(), APIC_DEVICE_CUDA);
        if (graph == nullptr) {
            throw std::runtime_error("Failed to load APIC graph '" + wrp_path + "': " + WarpErrorString());
        }

        const size_t output_bytes = output_element_count * sizeof(float);
        const size_t expected = wp_apic_get_param_size(graph, output_names[0].c_str());
        if (output_bytes != expected) {
            throw std::runtime_error(
                "Output '" + output_names[0] + "' byte size mismatch from output_shape: shape has " +
                std::to_string(output_bytes) + " bytes, APIC param expects " + std::to_string(expected));
        }
    }

    ~WrpRunnerKernel() {
        if (graph != nullptr) {
            wp_apic_destroy_graph(graph);
            graph = nullptr;
        }
    }

    std::vector<int64_t> GetInputShape(const OrtValue* value, size_t* element_count) const {
        OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        CheckOrt(g_ort->GetTensorTypeAndShape(value, &shape_info));

        ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        CheckOrt(g_ort->GetTensorElementType(shape_info, &dtype));
        if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);
            throw std::runtime_error("WrpRunner prototype only supports tensor(float)");
        }

        size_t rank = 0;
        CheckOrt(g_ort->GetDimensionsCount(shape_info, &rank));
        std::vector<int64_t> dims(rank);
        CheckOrt(g_ort->GetDimensions(shape_info, dims.data(), rank));
        CheckOrt(g_ort->GetTensorShapeElementCount(shape_info, element_count));
        g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);
        return dims;
    }

    void Compute(OrtKernelContext* ctx) {
        wp_cuda_context_set_current(context);

        for (int i = 0; i < kInputCount; ++i) {
            const OrtValue* input = nullptr;
            CheckOrt(g_ort->KernelContext_GetInput(ctx, i, &input));

            size_t element_count = 0;
            GetInputShape(input, &element_count);

            void* input_data = nullptr;
            CheckOrt(g_ort->GetTensorMutableData(const_cast<OrtValue*>(input), &input_data));
            const size_t byte_size = element_count * sizeof(float);
            const size_t expected = wp_apic_get_param_size(graph, input_names[i].c_str());
            if (byte_size != expected) {
                throw std::runtime_error(
                    "Input '" + input_names[i] + "' byte size mismatch: ONNX tensor has " +
                    std::to_string(byte_size) + " bytes, APIC param expects " + std::to_string(expected));
            }
            if (!wp_apic_set_param(graph, input_names[i].c_str(), input_data, byte_size)) {
                throw std::runtime_error("Failed to set APIC param '" + input_names[i] + "': " + WarpErrorString());
            }
        }

        if (!wp_apic_launch(graph, nullptr)) {
            throw std::runtime_error("Failed to launch APIC graph: " + WarpErrorString());
        }
        wp_cuda_context_synchronize(context);

        OrtValue* output = nullptr;
        CheckOrt(g_ort->KernelContext_GetOutput(
            ctx,
            0,
            output_shape.data(),
            output_shape.size(),
            &output));

        void* output_data = nullptr;
        CheckOrt(g_ort->GetTensorMutableData(output, &output_data));
        const size_t output_bytes = output_element_count * sizeof(float);
        const size_t expected = wp_apic_get_param_size(graph, output_names[0].c_str());
        if (output_bytes != expected) {
            throw std::runtime_error(
                "Output '" + output_names[0] + "' byte size mismatch: ONNX tensor has " +
                std::to_string(output_bytes) + " bytes, APIC param expects " + std::to_string(expected));
        }
        if (!wp_apic_get_param(graph, output_names[0].c_str(), output_data, output_bytes)) {
            throw std::runtime_error("Failed to get APIC param '" + output_names[0] + "': " + WarpErrorString());
        }
    }

    std::string wrp_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<int64_t> output_shape;
    size_t output_element_count = 0;
    void* context = nullptr;
    APICGraph graph = nullptr;
};

OrtStatusPtr CreateKernelV2(const OrtCustomOp*, const OrtApi*, const OrtKernelInfo* info, void** kernel) {
    try {
        *kernel = new WrpRunnerKernel(info);
        return nullptr;
    } catch (const std::exception& e) {
        return g_ort->CreateStatus(ORT_FAIL, e.what());
    }
}

void* CreateKernel(const OrtCustomOp*, const OrtApi*, const OrtKernelInfo* info) {
    try {
        return new WrpRunnerKernel(info);
    } catch (...) {
        return nullptr;
    }
}

void DestroyKernel(void* kernel) {
    delete static_cast<WrpRunnerKernel*>(kernel);
}

void KernelCompute(void* kernel, OrtKernelContext* context) {
    static_cast<WrpRunnerKernel*>(kernel)->Compute(context);
}

OrtStatusPtr KernelComputeV2(void* kernel, OrtKernelContext* context) {
    try {
        static_cast<WrpRunnerKernel*>(kernel)->Compute(context);
        return nullptr;
    } catch (const std::exception& e) {
        return g_ort->CreateStatus(ORT_FAIL, e.what());
    }
}

const char* GetName(const OrtCustomOp*) {
    return kOpName;
}

const char* GetExecutionProviderType(const OrtCustomOp*) {
    return nullptr;  // CPU EP. APIC handles its own CUDA launch internally.
}

size_t GetInputTypeCount(const OrtCustomOp*) {
    return kInputCount;
}

ONNXTensorElementDataType GetInputType(const OrtCustomOp*, size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t GetOutputTypeCount(const OrtCustomOp*) {
    return kOutputCount;
}

ONNXTensorElementDataType GetOutputType(const OrtCustomOp*, size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(const OrtCustomOp*, size_t) {
    return INPUT_OUTPUT_REQUIRED;
}

OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(const OrtCustomOp*, size_t) {
    return INPUT_OUTPUT_REQUIRED;
}

OrtMemType GetInputMemoryType(const OrtCustomOp*, size_t) {
    return OrtMemTypeDefault;
}

int GetVariadicInputMinArity(const OrtCustomOp*) {
    return 1;
}

int GetVariadicInputHomogeneity(const OrtCustomOp*) {
    return 1;
}

int GetVariadicOutputMinArity(const OrtCustomOp*) {
    return 1;
}

int GetVariadicOutputHomogeneity(const OrtCustomOp*) {
    return 1;
}

int GetStartVersion(const OrtCustomOp*) {
    return 1;
}

int GetEndVersion(const OrtCustomOp*) {
    return 1;
}

OrtCustomOp CreateWrpRunnerOp() {
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

OrtCustomOp g_wrp_runner_op = CreateWrpRunnerOp();

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    g_ort = api_base->GetApi(ORT_API_VERSION);

    OrtCustomOpDomain* domain = nullptr;
    OrtStatus* status = g_ort->CreateCustomOpDomain(kDomain, &domain);
    if (status != nullptr) {
        return status;
    }

    status = g_ort->CustomOpDomain_Add(domain, &g_wrp_runner_op);
    if (status != nullptr) {
        return status;
    }

    status = g_ort->AddCustomOpDomain(options, domain);
    if (status != nullptr) {
        return status;
    }

    return nullptr;
}
