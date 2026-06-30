#include "wrpb_archive.h"

#include <fstream>
#include <set>
#include <stdexcept>

namespace leapp::warp_runtime {
namespace {

std::uint32_t ReadU32(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    if (offset + 4 > size) {
        throw std::runtime_error("WRPB archive truncated while reading u32");
    }
    const std::uint8_t* p = data + offset;
    offset += 4;
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t ReadU64(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    if (offset + 8 > size) {
        throw std::runtime_error("WRPB archive truncated while reading u64");
    }
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(data[offset + i]) << (8 * i);
    }
    offset += 8;
    return value;
}

void ValidateRelativePath(const std::filesystem::path& rel) {
    if (rel.empty() || rel.is_absolute()) {
        throw std::runtime_error("WRPB entry path must be a non-empty relative path");
    }
    for (const auto& part : rel) {
        if (part == "..") {
            throw std::runtime_error("WRPB entry path escapes extraction directory");
        }
    }
}

}  // namespace

std::vector<WRPBEntry> ParseWRPB(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr || size < 12) {
        throw std::runtime_error("WRPB archive is empty or truncated");
    }
    if (!(data[0] == 'W' && data[1] == 'R' && data[2] == 'P' && data[3] == 'B')) {
        throw std::runtime_error("WRPB archive has bad magic");
    }

    std::size_t offset = 4;
    const std::uint32_t version = ReadU32(data, size, offset);
    if (version != 1) {
        throw std::runtime_error("Unsupported WRPB archive version: " + std::to_string(version));
    }

    const std::uint32_t entry_count = ReadU32(data, size, offset);
    std::vector<WRPBEntry> entries;
    entries.reserve(entry_count);
    std::set<std::string> seen_paths;

    for (std::uint32_t i = 0; i < entry_count; ++i) {
        const std::uint32_t path_len = ReadU32(data, size, offset);
        if (offset + path_len > size) {
            throw std::runtime_error("WRPB archive truncated while reading path");
        }
        std::string rel_path(reinterpret_cast<const char*>(data + offset), path_len);
        offset += path_len;
        ValidateRelativePath(rel_path);
        if (!seen_paths.insert(rel_path).second) {
            throw std::runtime_error("WRPB archive contains duplicate path: " + rel_path);
        }

        const std::uint64_t data_len = ReadU64(data, size, offset);
        if (offset + data_len > size) {
            throw std::runtime_error("WRPB archive truncated while reading entry data");
        }
        WRPBEntry entry;
        entry.relative_path = std::move(rel_path);
        entry.data.assign(data + offset, data + offset + static_cast<std::size_t>(data_len));
        offset += static_cast<std::size_t>(data_len);
        entries.push_back(std::move(entry));
    }

    if (offset != size) {
        throw std::runtime_error("WRPB archive contains trailing bytes");
    }
    return entries;
}

void ExtractWRPBToDirectory(const std::uint8_t* data,
                            std::size_t size,
                            const std::filesystem::path& dest_dir) {
    const auto entries = ParseWRPB(data, size);
    const auto base = std::filesystem::weakly_canonical(dest_dir);
    std::filesystem::create_directories(base);

    for (const auto& entry : entries) {
        const auto out_path = (base / entry.relative_path).lexically_normal();
        const auto parent = out_path.parent_path();
        const auto normalized = out_path.lexically_normal().string();
        const auto base_text = base.lexically_normal().string();
        if (normalized.compare(0, base_text.size(), base_text) != 0) {
            throw std::runtime_error("WRPB entry escapes extraction directory: " + entry.relative_path);
        }
        std::filesystem::create_directories(parent);
        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to write WRPB entry: " + out_path.string());
        }
        out.write(reinterpret_cast<const char*>(entry.data.data()),
                  static_cast<std::streamsize>(entry.data.size()));
    }
}

}  // namespace leapp::warp_runtime
