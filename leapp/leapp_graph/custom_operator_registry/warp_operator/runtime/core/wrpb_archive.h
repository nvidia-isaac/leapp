#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace leapp::warp_runtime {

struct WRPBEntry {
    std::string relative_path;
    std::vector<std::uint8_t> data;
};

std::vector<WRPBEntry> ParseWRPB(const std::uint8_t* data, std::size_t size);
void ExtractWRPBToDirectory(const std::uint8_t* data,
                            std::size_t size,
                            const std::filesystem::path& dest_dir);

}  // namespace leapp::warp_runtime
