// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include "torch/executor.h"

#include <ATen/core/grad_mode.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../bridge/cuda.h"
#include "torch/model.h"

namespace leapp_torch::native {

namespace cuda = leapp_native::cuda;
namespace adapter = leapp_native::tensors;
using adapter::View;

struct Call;
struct Outputs;

namespace {

constexpr std::array<at::ScalarType, 13> types = {
    at::kBool,  at::kByte,   at::kUInt16,   at::kUInt32, at::kUInt64,
    at::kChar,  at::kShort,  at::kInt,      at::kLong,   at::kHalf,
    at::kFloat, at::kDouble, at::kBFloat16,
};

at::Device device(int32_t id) {
  return id < 0 ? at::Device(at::kCPU) : at::Device(at::kCUDA, id);
}

at::Tensor borrow(const View& view) {
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
  return value;
}

bool same_tensor(const View& left, const View& right) {
  // Empty slots share sentinel addresses, not tensor identity.
  return left.bytes != 0 && left.data == right.data &&
         left.device == right.device && left.port.dtype == right.port.dtype &&
         at::IntArrayRef(left.port.shape, left.port.rank) ==
             at::IntArrayRef(right.port.shape, right.port.rank);
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
  std::unique_ptr<Outputs> run_dynamic(
      std::span<const View> inputs,
      std::span<const adapter::OutputBinding> outputs);

  at::Device target;

  cudaStream_t external_stream = nullptr;
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

  Context(const std::string& path, int32_t id, cudaStream_t external,
          std::span<const Port> in, std::span<const Port> out)
      : target(device(id)),
        external_stream(external),
        inputs(in.begin(), in.end()),
        outputs(out.begin(), out.end()) {
    if (id >= 0) {
      stream = std::make_unique<cuda::Stream>(id, external_stream);
    }
    Execution execution(*this);
    model = load_model(path, target, in, out);
    plan = model->plan;
    TORCH_CHECK(plan.access.size() == inputs.size(),
                "Torch input count mismatch");
    execution.finish();
  }

  explicit Context(const Context& source)
      : target(source.target),
        external_stream(source.external_stream),
        model(source.model),
        plan(source.plan),
        inputs(source.inputs),
        outputs(source.outputs) {
    if (target.is_cuda()) {
      stream = std::make_unique<cuda::Stream>(target.index(), external_stream);
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
    // The engine supplies validated inputs in the loaded port order.
    for (size_t i = 0; i < views.size(); ++i) {
      auto value = borrow(views[i]);
      sources.push_back(value);
      size_t alias = i;
      for (size_t j = 0; j < i; ++j) {
        if (same_tensor(views[i], views[j])) {
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

// Restore invocation-scoped arguments after execution, including failures.
// Prepared sources and wrappers stay alive independently of caller storage.
struct InputBindings {
  Inputs& inputs;
  std::span<const adapter::InputRoute> routes;
  size_t initialized = 0;

  ~InputBindings() {
    for (size_t i = 0; i < initialized; ++i) {
      const auto position = routes[i].position;
      inputs.args[position] = inputs.sources[inputs.aliases[position]];
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

  void run(std::span<const View> views,
           std::span<const adapter::InputRoute> routes,
           std::span<const adapter::OutputBinding> outputs);

  size_t input_alignment() const {
    return !context.plan.background_work &&
           std::all_of(context.plan.access.begin(), context.plan.access.end(),
                       [](InputAccess access) {
                         return access == InputAccess::Borrow;
                       });
  }
};

struct Outputs : adapter::Outputs {
  std::vector<at::Tensor> values;
};

std::unique_ptr<Call> Context::prepare(std::span<const View> inputs,
                                       std::span<const View> outputs) {
  Execution execution(*this);
  auto call = std::make_unique<Call>(*this, inputs, outputs);
  execution.finish();
  return call;
}

void Call::run(std::span<const View> views,
               std::span<const adapter::InputRoute> routes,
               std::span<const adapter::OutputBinding> outputs) {
  InputBindings bindings{inputs, routes};
  // Keep result owners alive until the stream finishes, including on
  // exceptions.
  std::vector<at::Tensor> values, targets;
  Context::Execution execution(context);
  for (const auto& output : outputs) {
    targets.push_back(borrow(output.view));
  }
  for (const auto& route : routes) {
    const auto previous = routes.first(bindings.initialized);
    const auto alias =
        std::find_if(previous.begin(), previous.end(), [&](const auto& other) {
          return same_tensor(views[route.position], views[other.position]);
        });
    inputs.args[route.position] =
        alias == previous.end() ? c10::IValue(borrow(views[route.position]))
                                : inputs.args[alias->position];
    ++bindings.initialized;
  }
  inputs.refresh();
  context.forward(inputs.args, values);
  for (size_t i = 0; i < values.size(); ++i) {
    const auto target =
        std::find_if(outputs.begin(), outputs.end(),
                     [i](const auto& output) { return output.position == i; });
    const auto& destination = target == outputs.end()
                                  ? destinations[i]
                                  : targets[target - outputs.begin()];
    destination.copy_(values[i], /*non_blocking=*/true);
  }
  execution.finish();
}

std::unique_ptr<Outputs> Context::run_dynamic(
    std::span<const View> views,
    std::span<const adapter::OutputBinding> bindings) {
  // Owners outlive the completion guard, also on exceptions.
  std::optional<Inputs> inputs;
  auto outputs = std::make_unique<Outputs>();
  std::vector<at::Tensor> targets;
  Execution execution(*this);
  inputs.emplace(*this, views);
  inputs->refresh();
  forward(inputs->args, outputs->values);
  for (const auto& binding : bindings) {
    targets.push_back(borrow(binding.view));
    const auto& value = outputs->values[binding.position];
    TORCH_CHECK(targets.back().sizes() == value.sizes(),
                "output buffer shape mismatch");
    targets.back().copy_(value, /*non_blocking=*/true);
  }
  // Raw Rust views cannot represent strides or Torch's lazy negative bit.
  // Materialize those before exposing storage; ordinary contiguous values stay
  // borrowed until Rust snapshots them before returning.
  for (size_t i = 0; i < outputs->values.size(); ++i) {
    if (std::none_of(bindings.begin(), bindings.end(), [i](const auto& output) {
          return output.position == i;
        })) {
      outputs->values[i] = outputs->values[i].resolve_neg().contiguous();
    }
  }
  execution.finish();
  return outputs;
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

}  // namespace

struct Executor::State {
  explicit State(std::unique_ptr<Context> source)
      : context(std::move(source)) {}

  std::unique_ptr<Context> context;
  std::unique_ptr<Call> call;
};

Executor::Executor(const std::string& path, int32_t device, void* stream,
                   std::span<const adapter::Port> inputs,
                   std::span<const adapter::Port> outputs) {
  state_ = std::make_unique<State>(
      std::make_unique<Context>(path, device, static_cast<cudaStream_t>(stream),
                                ports(inputs), ports(outputs)));
}

Executor::Executor(std::unique_ptr<State> state) : state_(std::move(state)) {}

Executor::~Executor() = default;

std::unique_ptr<Executor> Executor::new_session() const {
  return std::unique_ptr<Executor>(new Executor(
      std::make_unique<State>(std::make_unique<Context>(*state_->context))));
}

bool Executor::is_prepared() const noexcept { return state_->call != nullptr; }

size_t Executor::input_alignment() const noexcept {
  return state_->call ? state_->call->input_alignment() : 0;
}

void Executor::prepare(std::span<const adapter::View> inputs,
                       std::span<const adapter::View> outputs) {
  state_->call = state_->context->prepare(inputs, outputs);
}

void Executor::run(std::span<const adapter::View> inputs,
                   std::span<const adapter::InputRoute> routes,
                   std::span<const adapter::OutputBinding> outputs) {
  state_->call->run(inputs, routes, outputs);
}

std::unique_ptr<adapter::Outputs> Executor::run_dynamic(
    std::span<const adapter::View> inputs,
    std::span<const adapter::OutputBinding> outputs) {
  auto result = state_->context->run_dynamic(inputs, outputs);
  result->views.reserve(result->values.size());
  for (size_t i = 0; i < result->values.size(); ++i) {
    const auto& tensor = result->values[i];
    result->views.push_back(
        {{nullptr, 0, state_->context->outputs[i].dtype, tensor.sizes().data(),
          static_cast<size_t>(tensor.dim())},
         reinterpret_cast<uintptr_t>(tensor.data_ptr()),
         tensor.nbytes(),
         tensor.is_cuda() ? tensor.get_device() : -1});
  }
  return result;
}

}  // namespace leapp_torch::native
