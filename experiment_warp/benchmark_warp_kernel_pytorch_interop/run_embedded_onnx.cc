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

#include "onnxruntime_cxx_api.h"

namespace {

constexpr int kFeatureDim = 256;
constexpr int kDenseOutDim = 128;

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path artifacts_dir = "artifacts";
    std::filesystem::path model = "artifacts/embedded_pipeline.onnx";
    std::filesystem::path custom_op = "build/libbenchmark_wrp_onnx_custom_op.so";
    int batch_size = 4096;
    int warmup = 5;
    int iterations = 20;
    std::filesystem::path output;
};

double Ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void PrependExecutableDirToLdLibraryPath() {
    const auto exe_dir = std::filesystem::read_symlink("/proc/self/exe").parent_path().string();
    const char* old_path = std::getenv("LD_LIBRARY_PATH");
    const std::string next = old_path == nullptr || old_path[0] == '\0'
                                 ? exe_dir
                                 : exe_dir + ":" + old_path;
    setenv("LD_LIBRARY_PATH", next.c_str(), 1);
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
        } else if (arg == "--model") {
            options.model = require_value("--model");
        } else if (arg == "--custom-op") {
            options.custom_op = require_value("--custom-op");
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
        options.output = options.artifacts_dir / "embedded_onnx_output.bin";
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        const std::size_t input_count = static_cast<std::size_t>(options.batch_size) * kFeatureDim;
        const std::size_t output_count = static_cast<std::size_t>(options.batch_size) * kDenseOutDim;
        const auto input = ReadFloats(options.artifacts_dir / "input.bin", input_count);

        const auto startup_start = Clock::now();
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "warp_embedded_onnx");
        Ort::SessionOptions session_options;
        session_options.RegisterCustomOpsLibrary(options.custom_op.c_str());
        PrependExecutableDirToLdLibraryPath();
        OrtCUDAProviderOptions cuda_options{};
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, options.model.c_str(), session_options);
        const auto startup_end = Clock::now();

        Ort::MemoryInfo cpu_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> input_shape{options.batch_size, kFeatureDim};
        const char* input_names[] = {"input"};
        const char* output_names[] = {"final"};

        auto run_once = [&]() -> std::vector<float> {
            Ort::Value input_value = Ort::Value::CreateTensor<float>(
                cpu_memory,
                const_cast<float*>(input.data()),
                input.size(),
                input_shape.data(),
                input_shape.size());
            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_value,
                1,
                output_names,
                1);
            float* output_data = outputs.front().GetTensorMutableData<float>();
            return std::vector<float>(output_data, output_data + output_count);
        };

        for (int i = 0; i < options.warmup; ++i) {
            (void)run_once();
        }

        std::vector<double> samples;
        samples.reserve(options.iterations);
        std::vector<float> final_output;
        for (int i = 0; i < options.iterations; ++i) {
            const auto start = Clock::now();
            final_output = run_once();
            const auto end = Clock::now();
            samples.push_back(Ms(start, end));
        }
        WriteFloats(options.output, final_output);

        const double total = std::accumulate(samples.begin(), samples.end(), 0.0);
        const double mean = total / static_cast<double>(samples.size());
        const double checksum = std::accumulate(final_output.begin(), final_output.end(), 0.0);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "embedded_onnx_startup_ms=" << Ms(startup_start, startup_end) << "\n";
        std::cout << "embedded_onnx_avg_ms=" << mean << "\n";
        std::cout << std::setprecision(6) << "embedded_onnx_checksum=" << checksum << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
