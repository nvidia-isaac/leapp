#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "apic.h"
#include "onnxruntime_c_api.h"
#include "warp.h"

// Forward-declare the handful of CUDA runtime entry points we need. Avoid
// including <cuda_runtime_api.h> because warp.h already defines vector types
// such as float4. Symbols are provided by CUDA::cudart at link time.
extern "C" int cudaMemcpyAsync(void* dst, const void* src, size_t count, int kind, void* stream);
extern "C" const char* cudaGetErrorString(int error);

namespace {

const OrtApi* g_ort = nullptr;

constexpr const char* kDomain = "com.nvidia.warp";
constexpr const char* kOpName = "WrpRunner";
// Node inputs: data inputs first, then the embedded bundle as the last input.
constexpr int kDataInputCount = 2;
constexpr int kBundleInputIndex = kDataInputCount;
constexpr int kInputCount = kDataInputCount + 1;
constexpr int kOutputCount = 1;

constexpr int kCudaSuccess = 0;
constexpr int kCudaMemcpyDeviceToDevice = 3;

void CheckOrt(OrtStatus* status) {
    if (status == nullptr) {
        return;
    }
    std::string message = g_ort->GetErrorMessage(status);
    g_ort->ReleaseStatus(status);
    throw std::runtime_error(message);
}

void CheckCuda(int status, const std::string& what) {
    if (status != kCudaSuccess) {
        const char* msg = cudaGetErrorString(status);
        throw std::runtime_error(what + ": CUDA error " + std::to_string(status) +
                                 (msg != nullptr ? std::string(" (") + msg + ")" : std::string()));
    }
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
    // ORT reports size including the trailing null terminator for string attrs.
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

// -----------------------------------------------------------------------------
// Embedded bundle archive handling.
//
// Each node carries its APIC bundle (.wrp + _modules/) as a little-endian "WRPB"
// archive stored in a uint8 tensor initializer wired in as the node's last
// input. The op runs on CUDA EP, but the bundle input stays CPU-resident so we
// can read the archive bytes and extract the .wrp. Data inputs/outputs remain
// device-resident.
// -----------------------------------------------------------------------------

uint32_t ReadU32(const std::string& data, size_t& offset) {
    if (offset + 4 > data.size()) {
        throw std::runtime_error("Embedded bundle archive truncated (u32)");
    }
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data() + offset);
    offset += 4;
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t ReadU64(const std::string& data, size_t& offset) {
    if (offset + 8 > data.size()) {
        throw std::runtime_error("Embedded bundle archive truncated (u64)");
    }
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data() + offset);
    offset += 8;
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    return value;
}

std::string MakeTempDir() {
    namespace fs = std::filesystem;
    std::string tmpl = (fs::temp_directory_path() / "wrp_embedded_cuda_XXXXXX").string();
    std::vector<char> buffer(tmpl.begin(), tmpl.end());
    buffer.push_back('\0');
    char* created = mkdtemp(buffer.data());
    if (created == nullptr) {
        throw std::runtime_error("mkdtemp failed while extracting embedded .wrp bundle");
    }
    return std::string(created);
}

void ExtractArchive(const std::string& archive, const std::string& dest_dir) {
    namespace fs = std::filesystem;
    size_t offset = 0;

    if (archive.size() < 4 || archive.compare(0, 4, "WRPB") != 0) {
        throw std::runtime_error("Embedded bundle archive has bad magic (expected WRPB)");
    }
    offset = 4;

    const uint32_t version = ReadU32(archive, offset);
    if (version != 1) {
        throw std::runtime_error("Unsupported embedded bundle archive version: " +
                                 std::to_string(version));
    }

    const uint32_t num_entries = ReadU32(archive, offset);
    for (uint32_t i = 0; i < num_entries; ++i) {
        const uint32_t path_len = ReadU32(archive, offset);
        if (offset + path_len > archive.size()) {
            throw std::runtime_error("Embedded bundle archive truncated (path)");
        }
        const std::string rel_path = archive.substr(offset, path_len);
        offset += path_len;

        const uint64_t data_len = ReadU64(archive, offset);
        if (offset + data_len > archive.size()) {
            throw std::runtime_error("Embedded bundle archive truncated (data)");
        }

        const fs::path out_path = fs::path(dest_dir) / rel_path;
        // Guard against path escapes from a malformed archive.
        const fs::path normalized = out_path.lexically_normal();
        const fs::path base = fs::path(dest_dir).lexically_normal();
        if (normalized.string().compare(0, base.string().size(), base.string()) != 0) {
            throw std::runtime_error("Embedded bundle entry escapes temp dir: " + rel_path);
        }

        fs::create_directories(out_path.parent_path());
        std::ofstream ofs(out_path, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to write extracted bundle file: " + out_path.string());
        }
        ofs.write(archive.data() + offset, static_cast<std::streamsize>(data_len));
        offset += data_len;
    }
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
        wrp_name = GetStringAttribute(info, "wrp_name");
        input_names = SplitCsv(GetStringAttribute(info, "input_names"));
        output_names = SplitCsv(GetStringAttribute(info, "output_names"));
        output_shape = ParseShapeAttribute(GetStringAttribute(info, "output_shape"));
        output_element_count = ShapeElementCount(output_shape);

        if (input_names.size() != kDataInputCount) {
            throw std::runtime_error("WrpRunner prototype expects exactly 2 data input_names");
        }
        if (output_names.size() != kOutputCount) {
            throw std::runtime_error("WrpRunner prototype expects exactly 1 output_name");
        }

        EnsureWarpCudaInitialized(&context);
    }

    ~WrpRunnerKernel() {
        if (graph != nullptr) {
            wp_apic_destroy_graph(graph);
            graph = nullptr;
        }
        if (!temp_dir.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(temp_dir, ec);
        }
    }

    std::vector<int64_t> GetTensorShape(const OrtValue* value,
                                        ONNXTensorElementDataType* dtype,
                                        size_t* element_count) const {
        OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        CheckOrt(g_ort->GetTensorTypeAndShape(value, &shape_info));
        CheckOrt(g_ort->GetTensorElementType(shape_info, dtype));

        size_t rank = 0;
        CheckOrt(g_ort->GetDimensionsCount(shape_info, &rank));
        std::vector<int64_t> dims(rank);
        CheckOrt(g_ort->GetDimensions(shape_info, dims.data(), rank));
        CheckOrt(g_ort->GetTensorShapeElementCount(shape_info, element_count));
        g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);
        return dims;
    }

    // Read the bundle bytes from the CPU-resident last input, extract, and load
    // the APIC graph. Done once per kernel instance on first Compute, because
    // the initializer input (possibly external-data) is only available here.
    void EnsureGraphLoaded(OrtKernelContext* ctx) {
        std::call_once(load_once, [&]() {
            const OrtValue* bundle = nullptr;
            CheckOrt(g_ort->KernelContext_GetInput(ctx, kBundleInputIndex, &bundle));

            ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            size_t byte_count = 0;
            GetTensorShape(bundle, &dtype, &byte_count);
            if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
                throw std::runtime_error("WrpRunner bundle input must be tensor(uint8)");
            }
            if (byte_count == 0) {
                throw std::runtime_error("WrpRunner bundle input is empty");
            }

            void* bundle_data = nullptr;
            CheckOrt(g_ort->GetTensorMutableData(const_cast<OrtValue*>(bundle), &bundle_data));
            const std::string archive(static_cast<const char*>(bundle_data), byte_count);

            temp_dir = MakeTempDir();
            ExtractArchive(archive, temp_dir);
            const std::string wrp_full = (std::filesystem::path(temp_dir) / wrp_name).string();

            wp_cuda_context_set_current(context);
            graph = wp_apic_load_graph(context, wrp_full.c_str(), APIC_DEVICE_CUDA);
            if (graph == nullptr) {
                throw std::runtime_error("Failed to load embedded APIC graph '" + wrp_name +
                                         "': " + WarpErrorString());
            }

            const size_t output_bytes = output_element_count * sizeof(float);
            const size_t expected = wp_apic_get_param_size(graph, output_names[0].c_str());
            if (output_bytes != expected) {
                throw std::runtime_error(
                    "Output '" + output_names[0] +
                    "' byte size mismatch from output_shape: shape has " +
                    std::to_string(output_bytes) + " bytes, APIC param expects " +
                    std::to_string(expected));
            }
        });
    }

    void Compute(OrtKernelContext* ctx) {
        EnsureGraphLoaded(ctx);
        wp_cuda_context_set_current(context);

        // Run all data movement and the APIC launch on ORT's compute stream so
        // the node remains ordered with adjacent CUDA EP nodes without a full
        // device synchronization.
        void* stream = nullptr;
        CheckOrt(g_ort->KernelContext_GetGPUComputeStream(ctx, &stream));

        for (int i = 0; i < kDataInputCount; ++i) {
            const OrtValue* input = nullptr;
            CheckOrt(g_ort->KernelContext_GetInput(ctx, i, &input));

            ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            size_t element_count = 0;
            GetTensorShape(input, &dtype, &element_count);
            if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                throw std::runtime_error("WrpRunner prototype data inputs must be tensor(float)");
            }

            void* input_data = nullptr;  // device pointer (CUDA EP).
            CheckOrt(g_ort->GetTensorMutableData(const_cast<OrtValue*>(input), &input_data));
            const size_t byte_size = element_count * sizeof(float);
            const size_t expected = wp_apic_get_param_size(graph, input_names[i].c_str());
            if (byte_size != expected) {
                throw std::runtime_error(
                    "Input '" + input_names[i] + "' byte size mismatch: ONNX tensor has " +
                    std::to_string(byte_size) + " bytes, APIC param expects " + std::to_string(expected));
            }
            void* param_ptr = wp_apic_get_param_ptr(graph, input_names[i].c_str());
            if (param_ptr == nullptr) {
                throw std::runtime_error("Failed to get APIC param pointer '" + input_names[i] +
                                         "': " + WarpErrorString());
            }
            CheckCuda(cudaMemcpyAsync(param_ptr, input_data, byte_size, kCudaMemcpyDeviceToDevice, stream),
                      "device copy into APIC param '" + input_names[i] + "'");
        }

        if (!wp_apic_launch(graph, stream)) {
            throw std::runtime_error("Failed to launch APIC graph: " + WarpErrorString());
        }

        OrtValue* output = nullptr;
        CheckOrt(g_ort->KernelContext_GetOutput(
            ctx,
            0,
            output_shape.data(),
            output_shape.size(),
            &output));

        void* output_data = nullptr;  // device pointer (CUDA EP).
        CheckOrt(g_ort->GetTensorMutableData(output, &output_data));
        const size_t output_bytes = output_element_count * sizeof(float);
        const size_t expected = wp_apic_get_param_size(graph, output_names[0].c_str());
        if (output_bytes != expected) {
            throw std::runtime_error(
                "Output '" + output_names[0] + "' byte size mismatch: ONNX tensor has " +
                std::to_string(output_bytes) + " bytes, APIC param expects " + std::to_string(expected));
        }
        void* out_param_ptr = wp_apic_get_param_ptr(graph, output_names[0].c_str());
        if (out_param_ptr == nullptr) {
            throw std::runtime_error("Failed to get APIC param pointer '" + output_names[0] +
                                     "': " + WarpErrorString());
        }
        CheckCuda(cudaMemcpyAsync(output_data, out_param_ptr, output_bytes, kCudaMemcpyDeviceToDevice, stream),
                  "device copy from APIC param '" + output_names[0] + "'");
    }

    std::string wrp_name;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<int64_t> output_shape;
    size_t output_element_count = 0;
    std::string temp_dir;
    std::once_flag load_once;
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
    return "CUDAExecutionProvider";
}

size_t GetInputTypeCount(const OrtCustomOp*) {
    return kInputCount;
}

ONNXTensorElementDataType GetInputType(const OrtCustomOp*, size_t index) {
    // Data inputs are float; the trailing bundle input is uint8.
    if (index == static_cast<size_t>(kBundleInputIndex)) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }
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

OrtMemType GetInputMemoryType(const OrtCustomOp*, size_t index) {
    // The bundle initializer must stay on the host so we can read its bytes and
    // extract the .wrp; data inputs are device-resident on the CUDA EP.
    if (index == static_cast<size_t>(kBundleInputIndex)) {
        return OrtMemTypeCPUInput;
    }
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
