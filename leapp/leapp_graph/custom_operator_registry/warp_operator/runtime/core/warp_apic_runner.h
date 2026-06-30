#pragma once

#include "runtime_metadata.h"
#include "temp_bundle.h"
#include "tensor_view.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "apic.h"

namespace leapp::warp_runtime {

class WarpApicRunner {
 public:
    explicit WarpApicRunner(RuntimeMetadata metadata);
    ~WarpApicRunner();

    WarpApicRunner(const WarpApicRunner&) = delete;
    WarpApicRunner& operator=(const WarpApicRunner&) = delete;

    void LoadOnce(const std::uint8_t* bundle_data, std::size_t bundle_size);
    void Run(const RuntimeInvocation& invocation);

    const RuntimeMetadata& metadata() const { return metadata_; }

 private:
    void EnsureCudaInitialized();
    void ValidateInput(const TensorView& tensor, const InputSpec& spec) const;
    void ValidateOutput(const TensorView& tensor, const OutputSpec& spec) const;

    RuntimeMetadata metadata_;
    std::unique_ptr<TempBundleDir> temp_dir_;
    std::once_flag load_once_;
    std::mutex run_mutex_;
    void* cuda_context_ = nullptr;
    APICGraph* graph_ = nullptr;
};

}  // namespace leapp::warp_runtime
