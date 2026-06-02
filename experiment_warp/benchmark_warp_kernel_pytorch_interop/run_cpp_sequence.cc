#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "apic.h"
#include "onnxruntime_cxx_api.h"
#include "warp.h"

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr int cudaMemcpyDeviceToDevice = 3;
constexpr int cudaMemcpyDefault = 4;
extern "C" cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, int kind);
extern "C" cudaError_t cudaDeviceSynchronize();
extern "C" const char* cudaGetErrorString(cudaError_t error);

namespace {

constexpr int kFeatureDim = 256;
constexpr int kDenseOutDim = 128;

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path artifacts_dir = "artifacts";
    int batch_size = 4096;
    int warmup = 5;
    int iterations = 20;
    std::filesystem::path output;
};

double Ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void CheckCuda(cudaError_t status, const std::string& context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(context + ": " + cudaGetErrorString(status));
    }
}

void PrependExecutableDirToLdLibraryPath() {
    const auto exe_dir = std::filesystem::read_symlink("/proc/self/exe").parent_path().string();
    const char* old_path = std::getenv("LD_LIBRARY_PATH");
    const std::string next = old_path == nullptr || old_path[0] == '\0'
                                 ? exe_dir
                                 : exe_dir + ":" + old_path;
    setenv("LD_LIBRARY_PATH", next.c_str(), 1);
}

std::string WarpErrorString() {
    const char* error = wp_get_error_string();
    return error == nullptr || error[0] == '\0' ? "unknown Warp error" : error;
}

std::vector<float> ReadFloats(const std::filesystem::path& path, std::size_t count) {
    std::vector<float> values(count);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (in.gcount() != static_cast<std::streamsize>(values.size() * sizeof(float))) {
        throw std::runtime_error("Unexpected size for " + path.string());
    }
    return values;
}

void WriteFloats(const std::filesystem::path& path, const std::vector<float>& values) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

APICGraph LoadGraph(void* context, const std::filesystem::path& path) {
    APICGraph graph = wp_apic_load_graph(context, path.c_str(), APIC_DEVICE_CUDA);
    if (graph == nullptr) {
        throw std::runtime_error("Failed to load " + path.string() + ": " + WarpErrorString());
    }
    return graph;
}

void SetParam(APICGraph graph, const char* name, const void* data, std::size_t bytes) {
    const std::size_t expected = wp_apic_get_param_size(graph, name);
    if (expected != bytes) {
        throw std::runtime_error(std::string("Param size mismatch for ") + name);
    }
    if (!wp_apic_set_param(graph, name, data, bytes)) {
        throw std::runtime_error(std::string("Failed to set APIC param ") + name + ": " + WarpErrorString());
    }
}

void* ParamPtr(APICGraph graph, const char* name, std::size_t bytes) {
    const std::size_t expected = wp_apic_get_param_size(graph, name);
    if (expected != bytes) {
        throw std::runtime_error(std::string("Param size mismatch for ") + name);
    }
    void* ptr = wp_apic_get_param_ptr(graph, name);
    if (ptr == nullptr) {
        throw std::runtime_error(std::string("Failed to get APIC param pointer ") + name);
    }
    return ptr;
}

void Launch(APICGraph graph, void* context) {
    if (!wp_apic_launch(graph, nullptr)) {
        throw std::runtime_error("Failed to launch APIC graph: " + WarpErrorString());
    }
    wp_cuda_context_synchronize(context);
}

Options ParseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--artifacts-dir") {
            options.artifacts_dir = require_value("--artifacts-dir");
        } else if (arg == "--batch-size") {
            options.batch_size = std::stoi(require_value("--batch-size"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(require_value("--warmup"));
        } else if (arg == "--iterations") {
            options.iterations = std::stoi(require_value("--iterations"));
        } else if (arg == "--output") {
            options.output = require_value("--output");
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (options.output.empty()) {
        options.output = options.artifacts_dir / "cpp_sequence_output.bin";
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        const std::size_t feature_count = static_cast<std::size_t>(options.batch_size) * kFeatureDim;
        const std::size_t dense_count = static_cast<std::size_t>(options.batch_size) * kDenseOutDim;
        const std::size_t feature_bytes = feature_count * sizeof(float);
        const std::size_t dense_bytes = dense_count * sizeof(float);
        const auto input = ReadFloats(options.artifacts_dir / "input.bin", feature_count);

        const auto startup_start = Clock::now();
        if (wp_init(nullptr) != 0) {
            throw std::runtime_error("Failed to initialize Warp: " + WarpErrorString());
        }
        void* context = wp_cuda_device_get_primary_context(0);
        if (context == nullptr) {
            throw std::runtime_error("Failed to get CUDA primary context: " + WarpErrorString());
        }
        wp_cuda_context_set_current(context);

        APICGraph branch_wave = LoadGraph(context, options.artifacts_dir / "branch_wave_features.wrp");
        APICGraph branch_stencil = LoadGraph(context, options.artifacts_dir / "branch_stencil_features.wrp");
        APICGraph merge = LoadGraph(context, options.artifacts_dir / "merge_parallel_features.wrp");
        APICGraph final = LoadGraph(context, options.artifacts_dir / "final_postprocess.wrp");

        SetParam(branch_wave, "input", input.data(), feature_bytes);
        SetParam(branch_stencil, "input", input.data(), feature_bytes);

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "warp_cpp_sequence");
        Ort::SessionOptions session_options;
        PrependExecutableDirToLdLibraryPath();
        OrtCUDAProviderOptions cuda_options{};
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session dense_session(env, (options.artifacts_dir / "dense.onnx").c_str(), session_options);
        const auto startup_end = Clock::now();

        float* branch_wave_out = static_cast<float*>(ParamPtr(branch_wave, "output", feature_bytes));
        float* branch_stencil_out = static_cast<float*>(ParamPtr(branch_stencil, "output", feature_bytes));
        float* merge_a = static_cast<float*>(ParamPtr(merge, "branch_a", feature_bytes));
        float* merge_b = static_cast<float*>(ParamPtr(merge, "branch_b", feature_bytes));
        float* merged = static_cast<float*>(ParamPtr(merge, "merged", feature_bytes));
        float* final_input = static_cast<float*>(ParamPtr(final, "dense_out", dense_bytes));

        Ort::MemoryInfo cuda_memory("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
        std::array<int64_t, 2> merged_shape{options.batch_size, kFeatureDim};
        const char* dense_input_names[] = {"merged"};
        const char* dense_output_names[] = {"dense_out"};

        auto run_once = [&]() {
            Launch(branch_wave, context);
            Launch(branch_stencil, context);
            CheckCuda(cudaMemcpy(merge_a, branch_wave_out, feature_bytes, cudaMemcpyDeviceToDevice), "copy branch_wave");
            CheckCuda(cudaMemcpy(merge_b, branch_stencil_out, feature_bytes, cudaMemcpyDeviceToDevice), "copy branch_stencil");
            Launch(merge, context);

            Ort::Value dense_input = Ort::Value::CreateTensor<float>(
                cuda_memory,
                merged,
                feature_count,
                merged_shape.data(),
                merged_shape.size());
            auto dense_outputs = dense_session.Run(
                Ort::RunOptions{nullptr},
                dense_input_names,
                &dense_input,
                1,
                dense_output_names,
                1);
            float* dense_out = dense_outputs.front().GetTensorMutableData<float>();
            CheckCuda(cudaMemcpy(final_input, dense_out, dense_bytes, cudaMemcpyDefault), "copy dense_out");
            Launch(final, context);
        };

        for (int i = 0; i < options.warmup; ++i) {
            run_once();
        }

        std::vector<double> samples;
        samples.reserve(options.iterations);
        for (int i = 0; i < options.iterations; ++i) {
            const auto start = Clock::now();
            run_once();
            CheckCuda(cudaDeviceSynchronize(), "synchronize measured run");
            const auto end = Clock::now();
            samples.push_back(Ms(start, end));
        }

        std::vector<float> final_output(dense_count);
        if (!wp_apic_get_param(final, "final", final_output.data(), dense_bytes)) {
            throw std::runtime_error("Failed to read final output: " + WarpErrorString());
        }
        WriteFloats(options.output, final_output);

        const double total = std::accumulate(samples.begin(), samples.end(), 0.0);
        const double mean = total / static_cast<double>(samples.size());
        const double checksum = std::accumulate(final_output.begin(), final_output.end(), 0.0);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "cpp_sequence_startup_ms=" << Ms(startup_start, startup_end) << "\n";
        std::cout << "cpp_sequence_avg_ms=" << mean << "\n";
        std::cout << std::setprecision(6) << "cpp_sequence_checksum=" << checksum << "\n";

        wp_apic_destroy_graph(branch_wave);
        wp_apic_destroy_graph(branch_stencil);
        wp_apic_destroy_graph(merge);
        wp_apic_destroy_graph(final);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
