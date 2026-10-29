// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <ATen/core/grad_mode.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <utility>

#include "../bridge/api.h"
#include "../bridge/cuda.h"
#include "model.h"

namespace leapp_torch::native {

namespace cuda = leapp_native::cuda;
namespace adapter = leapp_native::bridge;
using adapter::View;

struct Call;
struct Outputs;

namespace {

constexpr std::array types{at::kBool,    at::kByte, at::kUInt16, at::kUInt32,
                           at::kUInt64,  at::kChar, at::kShort,  at::kInt,
                           at::kLong,    at::kHalf, at::kFloat,  at::kDouble,
                           at::kBFloat16};

at::Device device(int32_t id) {
  return id < 0 ? at::Device(at::kCPU) : at::Device(at::kCUDA, id);
}

at::Tensor borrow(const View& view) {
  TORCH_CHECK(view.port.dtype < types.size(), "unsupported Torch dtype");
  const auto shape = at::IntArrayRef(view.port.shape, view.port.rank);
  const auto options = at::TensorOptions()
                           .dtype(types[view.port.dtype])
                           .device(device(view.device));
  // The core already supplies validated placement; avoid querying CUDA pointer
  // attributes for each public output wrapper. Empty slots carry only metadata.
  auto value = view.bytes == 0
                   ? at::empty(shape, options)
                   : at::for_blob(reinterpret_cast<void*>(view.data), shape)
                         .options(options)
                         .target_device(device(view.device))
                         .make_tensor();
  TORCH_CHECK(value.nbytes() == view.bytes, "Torch tensor byte count mismatch");
  return value;
}

void check(const at::Tensor& value, const Port& port) {
  TORCH_CHECK(value.layout() == at::kStrided && !value.is_quantized(),
              "expected dense Torch tensor");
  TORCH_CHECK(
      port.dtype < types.size() && value.scalar_type() == types[port.dtype],
      "Torch tensor dtype mismatch");
  TORCH_CHECK(value.dim() == static_cast<int64_t>(port.shape.size()),
              "Torch tensor rank mismatch");
  for (size_t i = 0; i < port.shape.size(); ++i) {
    TORCH_CHECK(port.shape[i] == -1 || port.shape[i] == value.size(i),
                "Torch tensor shape mismatch");
  }
}

}  // namespace

at::ScalarType scalar_type(uint8_t dtype) {
  TORCH_CHECK(dtype < types.size(), "unsupported Torch dtype");
  return types[dtype];
}

struct Context {
  std::unique_ptr<Call> prepare(std::span<const View> inputs,
                                std::span<const View> outputs);
  std::unique_ptr<Outputs> run_dynamic(std::span<const View> inputs);

  at::Device target;

  std::unique_ptr<cuda::Stream> stream;
  std::shared_ptr<Model> model;
  InputPlan plan;
  std::vector<Port> inputs, outputs;

  // Keep this scope inside every operation that may enqueue native work.
  // It drains work before Rust can release its borrowed workspace/input
  // storage.
  struct Execution {
    Context& context;
    c10::DeviceGuard device_guard;
    at::NoGradGuard grad_guard;
    std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
    std::optional<cuda::Work> work;

    explicit Execution(Context& owner)
        : context(owner), device_guard(owner.target) {
      auto* stream = owner.stream.get();
      if (stream) {
        work.emplace(*stream);
        stream_guard.emplace(
            c10::cuda::getStreamFromExternal(stream->value, stream->device));
      }
    }

    void finish() {
      if (work) {
        work->finish(context.plan.background_work);
      }
    }
  };

  Context(const std::string& path, int32_t id, std::span<const Port> in,
          std::span<const Port> out)
      : target(device(id)),
        inputs(in.begin(), in.end()),
        outputs(out.begin(), out.end()) {
    if (id >= 0) {
      stream = std::make_unique<cuda::Stream>(id);
    }
    Execution execution(*this);
    model = load_model(path, target, in, out);
    plan = model->plan;
    TORCH_CHECK(plan.access.size() == inputs.size(),
                "Torch input count mismatch");
    execution.finish();
  }

  explicit Context(const Context& source)
      : target(source.target), model(source.model), plan(source.plan),
        inputs(source.inputs), outputs(source.outputs) {
    if (target.is_cuda()) {
      stream = std::make_unique<cuda::Stream>(target.index());
    }
  }

  void forward(const std::vector<c10::IValue>& args,
               std::vector<at::Tensor>& result) {
    // Fork bodies can run on worker streams. Publish prepared inputs before
    // dispatch, then complete their results before copying on this context's
    // stream.
    if (stream && plan.background_work) {
      C10_CUDA_CHECK(cudaDeviceSynchronize());
    }
    result = model->forward(args);
    if (stream && plan.background_work) {
      C10_CUDA_CHECK(cudaDeviceSynchronize());
    }
    TORCH_CHECK(result.size() == outputs.size(), "Torch output count mismatch");
    for (size_t i = 0; i < result.size(); ++i) {
      check(result[i], outputs[i]);
      TORCH_CHECK(result[i].device() == target, "Torch output device mismatch");
    }
  }
};

// Exact repeated inputs must still alias after mutation/retention isolation.
// Scratch owns storage separately from the TensorImpl handed to forward: a
// model may change its argument's shape or storage metadata in place.
struct Inputs {
  std::vector<at::Tensor> sources, scratch;
  std::vector<size_t> aliases;
  std::vector<InputAccess> access;
  std::vector<c10::IValue> args;

  Inputs(Context& context, std::span<const View> views)
      : access(context.plan.access) {
    TORCH_CHECK(views.size() == context.inputs.size(),
                "Torch input count mismatch");
    for (size_t i = 0; i < views.size(); ++i) {
      auto value = borrow(views[i]);
      check(value, context.inputs[i]);
      TORCH_CHECK(value.device() == context.target,
                  "Torch input device mismatch");
      sources.push_back(value);
      size_t alias = i;
      for (size_t j = 0; j < i; ++j) {
        // Empty slots share a sentinel address, not tensor identity.
        if (views[i].bytes != 0 && views[i].data == views[j].data &&
            value.scalar_type() == sources[j].scalar_type() &&
            value.sizes() == sources[j].sizes()) {
          alias = aliases[j];
          access[alias] = std::max(access[alias], access[i]);
          break;
        }
      }
      aliases.push_back(alias);
    }
    scratch.resize(views.size());
    args.resize(views.size());
    for (size_t i = 0; i < views.size(); ++i) {
      if (aliases[i] == i && access[i] == InputAccess::Scratch) {
        scratch[i] = at::empty_like(sources[i]);
      }
      if (access[aliases[i]] == InputAccess::Borrow) {
        args[i] = sources[aliases[i]];
      }
    }
  }

  void refresh() {
    for (size_t i = 0; i < sources.size(); ++i) {
      if (aliases[i] != i) {
        args[i] = args[aliases[i]];
      } else if (access[i] == InputAccess::Scratch) {
        scratch[i].copy_(sources[i]);
        args[i] = scratch[i].detach();
      } else if (access[i] == InputAccess::Owned) {
        args[i] = sources[i].clone();
      }
    }
  }
};

// Prepared calls retain TensorImpl/IValue metadata and fixed destination
// wrappers.
struct Call {
  Context& context;
  Inputs inputs;
  std::vector<at::Tensor> destinations;

  Call(Context& owner, std::span<const View> in, std::span<const View> out)
      : context(owner), inputs(owner, in) {
    TORCH_CHECK(out.size() == owner.outputs.size(),
                "Torch output count mismatch");
    for (size_t i = 0; i < out.size(); ++i) {
      auto value = borrow(out[i]);
      check(value, owner.outputs[i]);
      TORCH_CHECK(value.device() == owner.target,
                  "Torch output device mismatch");
      destinations.push_back(value);
    }
  }

  void run();
};

struct Outputs {
  void describe();

  std::vector<at::Tensor> values;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<adapter::View> views;
};

std::unique_ptr<Call> Context::prepare(std::span<const View> inputs,
                                       std::span<const View> outputs) {
  Execution execution(*this);
  auto call = std::make_unique<Call>(*this, inputs, outputs);
  execution.finish();
  return call;
}

void Call::run() {
  // Keep result owners alive until the stream finishes, including on
  // exceptions.
  std::vector<at::Tensor> values;
  Context::Execution execution(context);
  inputs.refresh();
  context.forward(inputs.args, values);
  for (size_t i = 0; i < values.size(); ++i) {
    TORCH_CHECK(values[i].sizes() == destinations[i].sizes(),
                "Torch output shape differs from fixed workspace");
  }
  for (size_t i = 0; i < values.size(); ++i) {
    destinations[i].copy_(values[i], /*non_blocking=*/true);
  }
  execution.finish();
}

std::unique_ptr<Outputs> Context::run_dynamic(std::span<const View> views) {
  // Owners outlive the completion guard, also on exceptions.
  std::optional<Inputs> inputs;
  auto outputs = std::make_unique<Outputs>();
  Execution execution(*this);
  inputs.emplace(*this, views);
  inputs->refresh();
  forward(inputs->args, outputs->values);
  // Raw Rust views cannot represent strides or Torch's lazy negative bit.
  // Materialize those before exposing storage; ordinary contiguous values stay
  // borrowed until Rust snapshots them before returning.
  for (auto& value : outputs->values) {
    value = value.resolve_neg().contiguous();
  }
  execution.finish();
  outputs->describe();
  return outputs;
}

void Outputs::describe() {
  shapes.reserve(values.size());
  views.reserve(values.size());
  for (const auto& value : values) {
    const auto type =
        std::find(types.begin(), types.end(), value.scalar_type());
    TORCH_CHECK(type != types.end(), "unsupported Torch output dtype");
    shapes.emplace_back(value.sizes().begin(), value.sizes().end());
    const auto& shape = shapes.back();
    views.push_back({{nullptr, 0, static_cast<uint8_t>(type - types.begin()),
                      shape.data(), shape.size()},
                     reinterpret_cast<uintptr_t>(value.data_ptr()),
                     value.nbytes(),
                     value.is_cuda() ? value.get_device() : -1});
  }
}

namespace {

Port own(const adapter::Port& port) {
  Port result{port.dtype, {}};
  if (port.rank) {
    result.shape.assign(port.shape, port.shape + port.rank);
  }
  return result;
}

std::vector<Port> ports(std::span<const adapter::Port> values) {
  std::vector<Port> result;
  result.reserve(values.size());
  for (const auto& value : values) {
    result.push_back(own(value));
  }
  return result;
}

// Load once. Peers share the model/weights and own their invocation resources.
// Rust holds a graph-wide guard through execution and output isolation.
struct Factory {
  Factory(const std::string& path, int32_t device,
          std::span<const adapter::Port> inputs,
          std::span<const adapter::Port> outputs)
      : context(std::make_unique<Context>(path, device, ports(inputs),
                                          ports(outputs))) {}

  bool requires_dynamic() const { return false; }

  std::unique_ptr<Context> create_session() {
    return std::make_unique<Context>(*context);
  }

  std::unique_ptr<Context> context;
};

template <typename F>
char* invoke(F&& function) noexcept {
  try {
    function();
    return nullptr;
  } catch (const std::exception& error) {
    auto* message = ::strdup(error.what());
    if (!message) {
      std::abort();
    }
    return message;
  } catch (...) {
    auto* message = ::strdup("unknown Torch failure");
    if (!message) {
      std::abort();
    }
    return message;
  }
}

const adapter::Api api{
    [](const char* path, int32_t device, const adapter::Port* inputs,
       size_t input_count, const adapter::Port* outputs, size_t output_count,
       void** value) noexcept {
      return invoke([&] {
        *value = new Factory(path, device, {inputs, input_count},
                             {outputs, output_count});
      });
    },
    [](void* value) noexcept { delete static_cast<Factory*>(value); },
    [](const void* value) noexcept {
      return static_cast<const Factory*>(value)->requires_dynamic();
    },
    [](void* model, void** value) noexcept {
      return invoke([&] {
        *value = static_cast<Factory*>(model)->create_session().release();
      });
    },
    [](void* value) noexcept { delete static_cast<Context*>(value); },
    [](void* session, const adapter::View* inputs, size_t input_count,
       const adapter::View* outputs, size_t output_count,
       void** value) noexcept {
      return invoke([&] {
        *value = static_cast<Context*>(session)
                     ->prepare({inputs, input_count}, {outputs, output_count})
                     .release();
      });
    },
    [](void* value) noexcept { delete static_cast<Call*>(value); },
    [](void* value) noexcept {
      return invoke([&] { static_cast<Call*>(value)->run(); });
    },
    [](void* session, const adapter::View* inputs, size_t count,
       void** value) noexcept {
      return invoke([&] {
        *value = static_cast<Context*>(session)
                     ->run_dynamic({inputs, count})
                     .release();
      });
    },
    [](void* value) noexcept { delete static_cast<Outputs*>(value); },
    [](const void* value, size_t* count) noexcept {
      const auto& views = static_cast<const Outputs*>(value)->views;
      *count = views.size();
      return views.data();
    },
    [](char* message) noexcept { std::free(message); }};

}  // namespace
}  // namespace leapp_torch::native

extern "C" __attribute__((visibility("default")))
const leapp_native::bridge::Api*
LEAPP_TORCH_ENTRY() noexcept {
  return &leapp_torch::native::api;
}
