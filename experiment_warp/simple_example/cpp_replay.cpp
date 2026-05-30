#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "apic.h"
#include "warp.h"

namespace {

struct Vec3 {
    float x;
    float y;
    float z;
};

void print_warp_error(const std::string& prefix) {
    const char* error = wp_get_error_string();
    std::cerr << prefix;
    if (error != nullptr && error[0] != '\0') {
        std::cerr << ": " << error;
    }
    std::cerr << '\n';
}

std::vector<Vec3> make_dummy_values(std::size_t byte_size, float scale) {
    const std::size_t count = byte_size / sizeof(Vec3);
    std::vector<Vec3> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float t = static_cast<float>(i % 1024) / 1024.0f;
        values[i] = Vec3{scale * (t + 0.1f), scale * (t + 0.2f), scale * (t + 0.3f)};
    }
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string graph_path = argc > 1 ? argv[1] : "warp_graph.wrp";

    if (wp_init(nullptr) != 0) {
        print_warp_error("Failed to initialize Warp");
        return EXIT_FAILURE;
    }

    if (!wp_is_cuda_enabled() || wp_cuda_device_get_count() == 0) {
        std::cerr << "This sample expects a CUDA-enabled Warp build and at least one CUDA device.\n";
        return EXIT_FAILURE;
    }

    void* context = wp_cuda_device_get_primary_context(0);
    if (context == nullptr) {
        print_warp_error("Failed to get CUDA primary context");
        return EXIT_FAILURE;
    }
    wp_cuda_context_set_current(context);

    APICGraph graph = wp_apic_load_graph(context, graph_path.c_str(), APIC_DEVICE_CUDA);
    if (graph == nullptr) {
        print_warp_error("Failed to load APIC graph " + graph_path);
        return EXIT_FAILURE;
    }

    const int num_params = wp_apic_get_num_params(graph);
    std::cout << "Loaded " << graph_path << " with " << num_params << " registered params:\n";
    for (int i = 0; i < num_params; ++i) {
        const char* name = wp_apic_get_param_name(graph, i);
        std::cout << "  " << name << " (" << wp_apic_get_param_size(graph, name) << " bytes)\n";
    }

    const std::size_t positions_bytes = wp_apic_get_param_size(graph, "positions");
    const std::size_t velocities_bytes = wp_apic_get_param_size(graph, "velocities");
    if (positions_bytes == 0 || velocities_bytes == 0) {
        std::cerr << "Expected graph params named 'positions' and 'velocities'.\n";
        wp_apic_destroy_graph(graph);
        return EXIT_FAILURE;
    }
    if (positions_bytes % sizeof(Vec3) != 0 || velocities_bytes % sizeof(Vec3) != 0) {
        std::cerr << "Expected positions/velocities to be tightly packed float3 buffers.\n";
        wp_apic_destroy_graph(graph);
        return EXIT_FAILURE;
    }

    auto positions = make_dummy_values(positions_bytes, 1.0f);
    auto velocities = make_dummy_values(velocities_bytes, 0.01f);

    if (!wp_apic_set_param(graph, "positions", positions.data(), positions_bytes)) {
        print_warp_error("Failed to set positions");
        wp_apic_destroy_graph(graph);
        return EXIT_FAILURE;
    }
    if (!wp_apic_set_param(graph, "velocities", velocities.data(), velocities_bytes)) {
        print_warp_error("Failed to set velocities");
        wp_apic_destroy_graph(graph);
        return EXIT_FAILURE;
    }

    if (!wp_apic_launch(graph, nullptr)) {
        print_warp_error("Failed to launch APIC graph");
        wp_apic_destroy_graph(graph);
        return EXIT_FAILURE;
    }
    wp_cuda_context_synchronize(context);

    std::vector<Vec3> out_positions(positions.size());
    if (!wp_apic_get_param(graph, "positions", out_positions.data(), positions_bytes)) {
        print_warp_error("Failed to read positions");
        wp_apic_destroy_graph(graph);
        return EXIT_FAILURE;
    }

    const std::size_t samples = std::min<std::size_t>(out_positions.size(), 5);
    std::cout << "First " << samples << " output positions:\n";
    for (std::size_t i = 0; i < samples; ++i) {
        const Vec3& p = out_positions[i];
        std::cout << "  [" << i << "] " << p.x << ", " << p.y << ", " << p.z << '\n';
    }

    wp_apic_destroy_graph(graph);
    return EXIT_SUCCESS;
}
