#pragma once

#include <filesystem>

namespace leapp::warp_runtime {

class TempBundleDir {
 public:
    TempBundleDir();
    ~TempBundleDir();

    TempBundleDir(const TempBundleDir&) = delete;
    TempBundleDir& operator=(const TempBundleDir&) = delete;

    const std::filesystem::path& path() const { return path_; }

 private:
    std::filesystem::path path_;
};

std::filesystem::path EphemeralExtractRoot();

}  // namespace leapp::warp_runtime
