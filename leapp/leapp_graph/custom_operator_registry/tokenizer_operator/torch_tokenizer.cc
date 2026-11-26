// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <torch/library.h>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

extern "C" {
struct TokenBatch { int64_t rows, width; const int64_t *ids, *mask, *types; };
const char* leapp_text_error();
void* leapp_tokenizer_create(const uint8_t*, size_t);
void leapp_tokenizer_destroy(void*);
TokenBatch* leapp_tokenize(const void*, const uint8_t*, const int64_t*, size_t, size_t, bool);
void leapp_token_batch_destroy(TokenBatch*);
}

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor> tokenize(
    const at::Tensor& data, const at::Tensor& lengths,
    const std::string& configuration, bool special) {
  TORCH_CHECK(data.device().is_cpu() && lengths.device().is_cpu(), "Tokenizer requires CPU tensors");
  TORCH_CHECK(data.scalar_type() == at::kByte && data.dim() == 2 &&
              lengths.scalar_type() == at::kLong && lengths.dim() == 1 &&
              data.size(0) == lengths.size(0) && data.size(0) > 0 && data.size(1) > 0,
              "Expected uint8[batch, capacity] bytes and int64[batch] lengths");
  static std::mutex mutex;
  static std::unordered_map<std::string, std::shared_ptr<void>> cache;
  std::shared_ptr<void> tokenizer;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(configuration);
    if (found == cache.end()) {
      auto pointer = leapp_tokenizer_create(
          reinterpret_cast<const uint8_t*>(configuration.data()), configuration.size());
      TORCH_CHECK(pointer, leapp_text_error());
      auto loaded = std::shared_ptr<void>(pointer, leapp_tokenizer_destroy);
      if (cache.size() >= 32) cache.clear();  // Active calls retain their handles.
      found = cache.emplace(configuration, std::move(loaded)).first;
    }
    tokenizer = found->second;
  }
  const auto bytes = data.contiguous(), sizes = lengths.contiguous();
  std::unique_ptr<TokenBatch, decltype(&leapp_token_batch_destroy)> result(
      leapp_tokenize(tokenizer.get(), bytes.data_ptr<uint8_t>(), sizes.data_ptr<int64_t>(),
                     data.size(0), data.size(1), special), leapp_token_batch_destroy);
  TORCH_CHECK(result, leapp_text_error());
  auto copy = [&](const int64_t* source) {
    auto output = at::empty({result->rows, result->width}, at::TensorOptions().dtype(at::kLong));
    if (output.numel()) std::memcpy(output.data_ptr<int64_t>(), source, output.nbytes());
    return output;
  };
  return {copy(result->ids), copy(result->mask), copy(result->types)};
}
}  // namespace

// Python owns the schema and fake implementation; this library supplies execution.
TORCH_LIBRARY_IMPL(leapp_text, CPU, m) { m.impl("tokenize", tokenize); }
