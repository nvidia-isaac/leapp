// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "leapp/native/export.h"

namespace lrt::native {

namespace {

[[noreturn]] void invalid(const char* message) {
  throw BackendError({ErrorCode::kInvalidArgument, "tensor", message});
}

}  // namespace

DLDataType dl_type(DataType dtype) {
  switch (dtype) {
    case DataType::kBool:
      return {kDLBool, 8, 1};
    case DataType::kUint8:
      return {kDLUInt, 8, 1};
    case DataType::kUint16:
      return {kDLUInt, 16, 1};
    case DataType::kUint32:
      return {kDLUInt, 32, 1};
    case DataType::kUint64:
      return {kDLUInt, 64, 1};
    case DataType::kInt8:
      return {kDLInt, 8, 1};
    case DataType::kInt16:
      return {kDLInt, 16, 1};
    case DataType::kInt32:
      return {kDLInt, 32, 1};
    case DataType::kInt64:
      return {kDLInt, 64, 1};
    case DataType::kFloat16:
      return {kDLFloat, 16, 1};
    case DataType::kBfloat16:
      return {kDLBfloat, 16, 1};
    case DataType::kFloat32:
      return {kDLFloat, 32, 1};
    case DataType::kFloat64:
      return {kDLFloat, 64, 1};
  }
  invalid("unsupported dtype");
}

DataType data_type(DLDataType dtype) {
  if (dtype.lanes != 1) {
    invalid("vector dtypes are unsupported");
  }
  for (auto type :
       {DataType::kBool, DataType::kUint8, DataType::kUint16, DataType::kUint32,
        DataType::kUint64, DataType::kInt8, DataType::kInt16, DataType::kInt32,
        DataType::kInt64, DataType::kFloat16, DataType::kBfloat16,
        DataType::kFloat32, DataType::kFloat64}) {
    const auto expected = dl_type(type);
    if (dtype.code == expected.code && dtype.bits == expected.bits) {
      return type;
    }
  }
  invalid("unsupported dtype");
}

size_t BorrowedTensor::bytes() const {
  const auto& value = descriptor();
  data_type(value.dtype);
  if (value.ndim < 0 || (value.ndim && !value.shape)) {
    invalid("invalid tensor rank/shape");
  }
  bool empty = false;
  for (auto dimension : shape()) {
    if (dimension < 0) {
      invalid("tensor dimensions must be concrete");
    }
    empty |= dimension == 0;
  }
  if (empty) {
    return 0;
  }
  size_t bytes = value.dtype.bits / 8;
  for (auto dimension : shape()) {
    if (static_cast<uint64_t>(dimension) >
        std::numeric_limits<size_t>::max() / bytes) {
      invalid("tensor byte size overflow");
    }
    bytes *= static_cast<size_t>(dimension);
  }
  return bytes;
}

namespace detail {

namespace {

void delete_message(void* value) noexcept {
  delete static_cast<std::string*>(value);
}

Error error(Status status) {
  if (status.ok()) {
    return {};
  }
  auto text = std::make_unique<std::string>(
      status.phase().empty() ? status.message()
                             : status.phase() + ": " + status.message());
  Error result{static_cast<int32_t>(status.code()), text->data(), text->size(),
               text.get(), delete_message};
  text.release();
  return result;
}

template <typename Function>
Error invoke(Function&& function) noexcept {
  try {
    return error(function());
  } catch (...) {
    try {
      throw;
    } catch (const BackendError& failure) {
      try {
        return error(failure.status());
      } catch (...) {
      }
    } catch (const std::exception& failure) {
      try {
        return error({ErrorCode::kInternal, "backend", failure.what()});
      } catch (...) {
      }
    } catch (...) {
    }
    // An allocation failure must not throw across the boundary or hide an
    // error.
    static constexpr char message[] = "unhandled native backend failure";
    static const std::string_view fallback(message, sizeof(message) - 1);
    return {static_cast<int32_t>(ErrorCode::kInternal), fallback.data(),
            fallback.size(), nullptr, nullptr};
  }
}

std::vector<PortSpec> ports(const Port* values, size_t count) {
  std::vector<PortSpec> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const auto& value = values[i];
    PortSpec port{
        std::string(value.name, value.name_size), data_type(value.dtype), {}};
    if (value.rank) {
      port.shape.assign(value.shape, value.shape + value.rank);
    }
    result.push_back(std::move(port));
  }
  return result;
}

std::vector<BorrowedTensor> inputs(const DLTensor* values, size_t count) {
  std::vector<BorrowedTensor> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    result.emplace_back(values[i]);
  }
  return result;
}

std::vector<::lrt::native::OutputBinding> bindings(const OutputBinding* values,
                                                   size_t count) {
  std::vector<::lrt::native::OutputBinding> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    result.push_back({values[i].position, WritableTensor(values[i].tensor)});
  }
  return result;
}

}  // namespace

struct Access {
  static Transfers transfers(const Host& host) { return Transfers(host); }

  static TensorHandle take(OwnedTensor& value) noexcept {
    return std::exchange(value.handle_, {});
  }
};

Error load(LoadBackend loader, const LoadArgs* args, void** output) noexcept {
  return invoke([&] {
    const auto in = ports(args->inputs, args->input_count);
    const auto out = ports(args->outputs, args->output_count);
    const LoadContext context{
        {args->path, args->path_size}, args->device, args->stream, in, out};
    std::unique_ptr<BackendPlugin> plugin;
    auto status = loader(context, &plugin);
    if (!status.ok()) {
      return status;
    }
    if (!plugin) {
      return Status{ErrorCode::kInternal, "load", "backend returned no plugin"};
    }
    *output = plugin.release();
    return Status{};
  });
}

Api make_api(decltype(Api::load) loader) noexcept {
  return {
      {1, sizeof(Api)},
      loader,
      [](void* value) noexcept { delete static_cast<BackendPlugin*>(value); },
      [](const void* value, void** output) noexcept {
        return invoke([&] {
          std::unique_ptr<BackendPlugin> peer;
          auto status =
              static_cast<const BackendPlugin*>(value)->new_session(&peer);
          if (!status.ok()) {
            return status;
          }
          if (!peer || peer->is_prepared()) {
            return Status{ErrorCode::kInternal, "new_session",
                          "backend returned no unprepared peer"};
          }
          *output = peer.release();
          return Status{};
        });
      },
      [](const void* value) noexcept {
        return static_cast<const BackendPlugin*>(value)->needs_guard();
      },
      [](const void* value) noexcept {
        return static_cast<const BackendPlugin*>(value)->can_prepare();
      },
      [](const void* value) noexcept {
        return static_cast<const BackendPlugin*>(value)->is_prepared();
      },
      [](const void* value) noexcept {
        return static_cast<const BackendPlugin*>(value)->input_alignment();
      },
      [](const void* value) noexcept {
        return static_cast<const BackendPlugin*>(value)->output_alignment();
      },
      [](void* value, const DLTensor* in, size_t in_count, const DLTensor* out,
         size_t out_count) noexcept {
        return invoke([&] {
          const auto input = inputs(in, in_count);
          std::vector<WritableTensor> output;
          output.reserve(out_count);
          for (size_t i = 0; i < out_count; ++i) {
            output.emplace_back(out[i]);
          }
          auto* plugin = static_cast<BackendPlugin*>(value);
          auto status = plugin->prepare(input, output);
          if (status.ok() && !plugin->is_prepared()) {
            return Status{ErrorCode::kInternal, "prepare",
                          "successful preparation left backend unprepared"};
          }
          return status;
        });
      },
      [](void* value, const NamedTensor* in, size_t in_count,
         const InputRoute* routes, size_t route_count, const OutputBinding* out,
         size_t out_count) noexcept {
        return invoke([&] {
          std::vector<::lrt::native::NamedTensor> input;
          input.reserve(in_count);
          for (size_t i = 0; i < in_count; ++i) {
            input.push_back(
                {{in[i].name, in[i].name_size}, BorrowedTensor(in[i].tensor)});
          }
          const auto output = bindings(out, out_count);
          return static_cast<BackendPlugin*>(value)->run(
              input, {routes, route_count}, output);
        });
      },
      [](void* value, const DLTensor* in, size_t in_count,
         const OutputBinding* out, size_t out_count, const Host* host,
         void** output) noexcept {
        return invoke([&] {
          const auto input = inputs(in, in_count);
          const auto targets = bindings(out, out_count);
          auto transfers = Access::transfers(*host);
          auto result = std::make_unique<DynamicOutputs>();
          auto status = static_cast<BackendPlugin*>(value)->run_dynamic(
              input, targets, transfers, result.get());
          if (!status.ok()) {
            return status;
          }
          if (std::any_of(
                  result->begin(), result->end(), [](const auto& tensor) {
                    return tensor.has_value() && !static_cast<bool>(*tensor);
                  })) {
            return Status{ErrorCode::kInternal, "execute",
                          "backend returned an empty owned tensor"};
          }
          *output = result.release();
          return Status{};
        });
      },
      [](const void* value) noexcept {
        return static_cast<const DynamicOutputs*>(value)->size();
      },
      [](void* value, size_t position) noexcept {
        auto& result = (*static_cast<DynamicOutputs*>(value))[position];
        return result ? Access::take(*result) : TensorHandle{};
      },
      [](void* value) noexcept { delete static_cast<DynamicOutputs*>(value); }};
}

}  // namespace detail

Status Transfers::copy(BorrowedTensor source, OwnedTensor* destination) {
  if (!destination || *destination) {
    return {ErrorCode::kInvalidArgument, "copy", "destination must be empty"};
  }
  OwnedTensor candidate;
  auto failure =
      host_.copy(host_.context, &source.descriptor(), &candidate.handle_);

  struct ErrorOwner {
    detail::Error& value;

    ~ErrorOwner() {
      if (value.destroy) {
        value.destroy(value.owner);
      }
    }
  } owner{failure};

  if (failure.code) {
    const auto code = failure.code >= 1 && failure.code <= 5
                          ? static_cast<ErrorCode>(failure.code)
                          : ErrorCode::kInternal;
    return {code, "copy", std::string(failure.message, failure.message_size)};
  }
  if (!candidate || !candidate.handle_.destroy) {
    return {ErrorCode::kInternal, "copy",
            "runtime returned an invalid tensor owner"};
  }
  *destination = std::move(candidate);
  return {};
}

}  // namespace lrt::native
