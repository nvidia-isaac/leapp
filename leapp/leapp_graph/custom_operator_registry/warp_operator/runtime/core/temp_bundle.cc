#include "temp_bundle.h"

#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace leapp::warp_runtime {

std::filesystem::path EphemeralExtractRoot() {
    namespace fs = std::filesystem;
    if (const char* override_dir = std::getenv("WARP_APIC_TMPDIR");
        override_dir != nullptr && override_dir[0] != '\0') {
        return fs::path(override_dir);
    }
#ifndef _WIN32
    if (fs::exists("/dev/shm") && fs::is_directory("/dev/shm")) {
        return "/dev/shm";
    }
    if (const char* tmpdir = std::getenv("TMPDIR"); tmpdir != nullptr && tmpdir[0] != '\0') {
        return fs::path(tmpdir);
    }
#endif
    return fs::temp_directory_path();
}

TempBundleDir::TempBundleDir() {
    namespace fs = std::filesystem;
    const fs::path root = EphemeralExtractRoot();
#ifdef _WIN32
    std::random_device rd;
    std::uniform_int_distribution<unsigned long long> dist;
    for (int attempt = 0; attempt < 64; ++attempt) {
        fs::path candidate = root / ("leapp_wrp_" + std::to_string(dist(rd)));
        std::error_code ec;
        if (fs::create_directory(candidate, ec) && !ec) {
            path_ = std::move(candidate);
            return;
        }
    }
    throw std::runtime_error("Failed to create Warp APIC temp directory");
#else
    std::string tmpl = (root / "leapp_wrp_XXXXXX").string();
    std::vector<char> buffer(tmpl.begin(), tmpl.end());
    buffer.push_back('\0');
    char* created = mkdtemp(buffer.data());
    if (created == nullptr) {
        throw std::runtime_error("mkdtemp failed while creating Warp APIC temp directory");
    }
    path_ = created;
#endif
}

TempBundleDir::~TempBundleDir() {
    if (!path_.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
}

}  // namespace leapp::warp_runtime
