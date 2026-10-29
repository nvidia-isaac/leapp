// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../bridge/api.h"
#include "../bridge/cuda.h"

static_assert(NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 16,
              "Build this adapter with TensorRT 10.16 or newer 10.x headers");

namespace leapp_tensorrt {

namespace {

namespace trt = nvinfer1;
namespace adapter = leapp_native::bridge;
namespace cuda = leapp_native::cuda;

using cuda::DeviceScope;
using cuda::Stream;
using cuda::Work;

thread_local std::string diagnostic;

class Logger : public trt::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      try {
        diagnostic = message;
      } catch (...) {
      }
    }
  }
};

Logger logger;

void require(bool success, std::string_view message) {
  if (!success) {
    throw std::runtime_error(std::string(message) +
                             (diagnostic.empty() ? "" : ": " + diagnostic));
  }
}

void invalid(bool valid, std::string_view message) {
  if (!valid) {
    throw std::runtime_error("invalid argument: TensorRT " +
                             std::string(message));
  }
}

void supported(bool valid, std::string_view message) {
  if (!valid) {
    throw std::runtime_error("backend unsupported: TensorRT " +
                             std::string(message));
  }
}

template <typename T>
struct DeviceDelete {
  int device;

  void operator()(T* value) const noexcept {
    DeviceScope scope(device);
    delete value;
  }
};

template <typename T>
using Owned = std::unique_ptr<T, DeviceDelete<T>>;

class Buffer {
 public:
  Buffer(int device, bool host) : device_(device), host_(host) {}

  ~Buffer() {
    DeviceScope scope(device_);
    release();
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  void reserve(uint64_t size, uint64_t alignment = 256) {
    size = std::max<uint64_t>(size, 1);
    alignment = std::max<uint64_t>(alignment, 256);
    invalid((alignment & (alignment - 1)) == 0 &&
                size <= std::numeric_limits<size_t>::max() - (alignment - 1),
            "allocation size or alignment overflow");
    if (data && capacity >= size &&
        reinterpret_cast<uintptr_t>(data) % alignment == 0) {
      return;
    }
    DeviceScope scope(device_);
    scope.check();
    void* memory = nullptr;
    const auto bytes = static_cast<size_t>(size + alignment - 1);
    cuda::check(host_ ? cudaMallocHost(&memory, bytes)
                      : cudaMalloc(&memory, bytes));
    release();
    base_ = memory;
    data = reinterpret_cast<void*>(
        (reinterpret_cast<uintptr_t>(memory) + alignment - 1) &
        ~(static_cast<uintptr_t>(alignment) - 1));
    capacity = size;
  }

  void* data = nullptr;
  uint64_t capacity = 0;

 private:
  void release() noexcept {
    if (base_) {
      if (host_) {
        cudaFreeHost(base_);
      } else {
        cudaFree(base_);
      }
    }
  }

  int device_;
  bool host_;
  void* base_ = nullptr;
};

struct Type {
  uint8_t code;
  size_t bytes;
};

Type type(trt::DataType value) {
  switch (value) {
    case trt::DataType::kBOOL:
      return {0, 1};
    case trt::DataType::kUINT8:
      return {1, 1};
    case trt::DataType::kINT8:
      return {5, 1};
    case trt::DataType::kINT32:
      return {7, 4};
    case trt::DataType::kINT64:
      return {8, 8};
    case trt::DataType::kHALF:
      return {9, 2};
    case trt::DataType::kFLOAT:
      return {10, 4};
    case trt::DataType::kBF16:
      return {12, 2};
    default:
      supported(false, "tensor dtype is not representable by the public API");
      return {};
  }
}

trt::Dims dimensions(const int64_t* shape, size_t rank) {
  invalid(rank <= trt::Dims::MAX_DIMS, "tensor rank exceeds the engine limit");
  trt::Dims result{};
  result.nbDims = static_cast<int32_t>(rank);
  std::copy_n(shape, rank, result.d);
  return result;
}

bool concrete(const trt::Dims& shape) {
  return shape.nbDims >= 0 && std::all_of(shape.d, shape.d + shape.nbDims,
                                          [](int64_t dim) { return dim >= 0; });
}

size_t byte_size(const trt::Dims& shape, size_t width) {
  invalid(concrete(shape), "tensor dimensions are unresolved");
  if (std::find(shape.d, shape.d + shape.nbDims, 0) != shape.d + shape.nbDims) {
    return 0;
  }
  size_t size = width;
  for (int i = 0; i < shape.nbDims; ++i) {
    invalid(
        static_cast<uint64_t>(shape.d[i]) <=
            static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) / size,
        "tensor byte size overflow");
    size *= static_cast<size_t>(shape.d[i]);
  }
  return size;
}

struct Bounds {
  trt::Dims minimum, maximum;
  std::vector<int64_t> value_minimum, value_maximum;
};

struct Port {
  std::string name;
  Type dtype;
  trt::Dims declared, engine_shape;
  bool host, control;
  std::vector<Bounds> bounds;

  void check(const trt::Dims& shape) const {
    invalid(shape.nbDims == declared.nbDims,
            "tensor rank mismatch for " + name);
    for (int i = 0; i < shape.nbDims; ++i) {
      invalid(shape.d[i] >= 0 &&
                  (declared.d[i] == -1 || declared.d[i] == shape.d[i]),
              "tensor shape mismatch for " + name);
    }
  }

  trt::Dims check(const adapter::View& view, int device) const {
    const auto shape = dimensions(view.port.shape, view.port.rank);
    check(shape);
    invalid(view.port.dtype == dtype.code && view.device == device,
            "tensor dtype or device mismatch for " + name);
    return shape;
  }
};

std::vector<int64_t> shape_bounds(const trt::ICudaEngine& engine,
                                  const Port& port, int profile,
                                  trt::OptProfileSelector selector) {
  const auto count = byte_size(port.engine_shape, 1);
  if (count == 0) {
    return {};
  }
  if (const auto* values = engine.getProfileTensorValuesV2(port.name.c_str(),
                                                           profile, selector)) {
    return {values, values + count};
  }
  const auto* values =
      engine.getProfileTensorValues(port.name.c_str(), profile, selector);
  require(values != nullptr,
          "TensorRT engine has no shape-value bounds for " + port.name);
  return {values, values + count};
}

struct Session;
struct Call;

struct Model {
  explicit Model(const char* path, int id, cudaStream_t stream,
                 std::span<const adapter::Port> in,
                 std::span<const adapter::Port> out)
      : device(id), external_stream(stream),
        runtime(nullptr, {id}), engine(nullptr, {id}) {
    supported(id >= 0, "requires CUDA execution");
    DeviceScope scope(device);
    scope.check();
    static std::once_flag plugins;
    std::call_once(plugins, [] {
      require(initLibNvInferPlugins(&logger, ""),
              "TensorRT plugin initialization failed");
    });
    runtime.reset(trt::createInferRuntime(logger));
    require(runtime != nullptr, "TensorRT runtime creation failed");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    invalid(file.is_open(), "cannot open engine file");
    const auto length = file.tellg();
    invalid(length > 0, "engine file is empty or unreadable");
    std::vector<char> data(static_cast<size_t>(length));
    file.seekg(0);
    invalid(static_cast<bool>(file.read(
                data.data(), static_cast<std::streamsize>(data.size()))),
            "cannot read engine file");
    engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
    require(engine != nullptr, "TensorRT engine deserialization failed");
    invalid(
        static_cast<size_t>(engine->getNbIOTensors()) == in.size() + out.size(),
        "bundle and engine port counts differ");
    std::set<std::string> names;
    inputs = ports(in, trt::TensorIOMode::kINPUT, names);
    outputs = ports(out, trt::TensorIOMode::kOUTPUT, names);
    dynamic = std::any_of(inputs.begin(), inputs.end(),
                          [](const Port& p) { return p.control; });
    shape_outputs = std::any_of(outputs.begin(), outputs.end(),
                                [](const Port& p) { return p.control; });
    inspect_fixed_shapes();
  }

  std::vector<Port> ports(std::span<const adapter::Port> values,
                          trt::TensorIOMode mode,
                          std::set<std::string>& names) {
    std::vector<Port> result;
    for (const auto& value : values) {
      std::string name(value.name, value.name_size);
      invalid(!name.empty() && name.size() < 4096 &&
                  name.find('\0') == std::string::npos &&
                  names.insert(name).second,
              "bundle port names must be unique engine tensor names");
      invalid(engine->getTensorIOMode(name.c_str()) == mode,
              "port name/direction mismatch for " + name);
      const auto dtype = type(engine->getTensorDataType(name.c_str()));
      invalid(value.dtype == dtype.code, "tensor dtype mismatch for " + name);
      invalid(value.rank <= trt::Dims::MAX_DIMS,
              "tensor rank exceeds the engine limit");
      const auto shape = engine->getTensorShape(name.c_str());
      invalid(shape.nbDims == static_cast<int32_t>(value.rank),
              "tensor rank mismatch for " + name);
      for (size_t i = 0; i < value.rank; ++i) {
        invalid(
            value.shape[i] >= -1 && (shape.d[i] == -1 || value.shape[i] == -1 ||
                                     shape.d[i] == value.shape[i]),
            "tensor shape mismatch for " + name);
      }
      supported(engine->getAliasedInputTensor(name.c_str()) == nullptr,
                "input/output aliasing is not supported for " + name);
      Port port{
          name,
          dtype,
          dimensions(value.shape, value.rank),
          shape,
          engine->getTensorLocation(name.c_str()) == trt::TensorLocation::kHOST,
          engine->isShapeInferenceIO(name.c_str()),
          {}};
      for (int profile = 0; profile < engine->getNbOptimizationProfiles();
           ++profile) {
        supported(
            engine->getTensorFormat(name.c_str(), profile) ==
                    trt::TensorFormat::kLINEAR &&
                engine->getTensorVectorizedDim(name.c_str(), profile) == -1,
            "only linear, unvectorized I/O is supported for " + name);
        if (mode == trt::TensorIOMode::kINPUT) {
          Bounds bound{engine->getProfileShape(name.c_str(), profile,
                                               trt::OptProfileSelector::kMIN),
                       engine->getProfileShape(name.c_str(), profile,
                                               trt::OptProfileSelector::kMAX),
                       {},
                       {}};
          require(concrete(bound.minimum) && concrete(bound.maximum) &&
                      bound.minimum.nbDims == shape.nbDims &&
                      bound.maximum.nbDims == shape.nbDims,
                  "TensorRT engine has invalid profile bounds");
          if (port.control) {
            supported(concrete(shape) && (dtype.code == 7 || dtype.code == 8),
                      "shape controls require fixed-size int32/int64 tensors");
            bound.value_minimum = shape_bounds(*engine, port, profile,
                                               trt::OptProfileSelector::kMIN);
            bound.value_maximum = shape_bounds(*engine, port, profile,
                                               trt::OptProfileSelector::kMAX);
          }
          port.bounds.push_back(std::move(bound));
        }
      }
      result.push_back(std::move(port));
    }
    return result;
  }

  int select_profile(
      std::span<const adapter::View> views,
      const std::vector<std::unique_ptr<Buffer>>& controls) const {
    for (int profile = 0; profile < engine->getNbOptimizationProfiles();
         ++profile) {
      bool matches = true;
      for (size_t i = 0; i < inputs.size() && matches; ++i) {
        const auto& port = inputs[i];
        const auto& bound = port.bounds[profile];
        const auto& view = views[i];
        for (size_t d = 0; d < view.port.rank; ++d) {
          matches &= view.port.shape[d] >= bound.minimum.d[d] &&
                     view.port.shape[d] <= bound.maximum.d[d];
        }
        for (size_t v = 0; v < bound.value_minimum.size() && matches; ++v) {
          const auto* data = controls[i]->data;
          const int64_t value = port.dtype.code == 7
                                    ? static_cast<const int32_t*>(data)[v]
                                    : static_cast<const int64_t*>(data)[v];
          matches &= value >= bound.value_minimum[v] &&
                     value <= bound.value_maximum[v];
        }
      }
      if (matches) {
        return profile;
      }
    }
    invalid(false,
            "no optimization profile accepts all input dimensions and shape "
            "values");
    return -1;
  }

  void inspect_fixed_shapes();

  bool requires_dynamic() const { return dynamic; }

  std::unique_ptr<Session> create_session();

  int device;
  cudaStream_t external_stream;
  Owned<trt::IRuntime> runtime;
  Owned<trt::ICudaEngine> engine;
  std::vector<Port> inputs, outputs;
  bool dynamic = false, shape_outputs = false;
};

struct Transfer {
  void* destination;
  const void* source;
  size_t bytes;
  cudaMemcpyKind kind;

  void run(cudaStream_t stream) const {
    if (bytes) {
      cuda::check(cudaMemcpyAsync(destination, source, bytes, kind, stream));
    }
  }
};

struct Execution {
  Execution(Model& owner, Stream& queue,
            trt::ExecutionContextAllocationStrategy strategy =
                trt::ExecutionContextAllocationStrategy::kSTATIC)
      : model(owner),
        stream(queue),
        staging(owner.inputs.size()),
        controls(owner.inputs.size()),
        addresses(owner.inputs.size()),
        context(nullptr, {owner.device}) {
    DeviceScope scope(model.device);
    scope.check();
    context.reset(model.engine->createExecutionContext(strategy));
    require(context != nullptr, "TensorRT context creation failed");
  }

  void inputs(std::span<const adapter::View> views) {
    invalid(views.size() == model.inputs.size(), "input count mismatch");
    transfers.clear();
    for (size_t i = 0; i < views.size(); ++i) {
      const auto& view = views[i];
      const auto& port = model.inputs[i];
      port.check(view, model.device);
      auto* source = reinterpret_cast<void*>(view.data);
      if (port.control) {
        if (!controls[i]) {
          controls[i] = std::make_unique<Buffer>(model.device, true);
        }
        controls[i]->reserve(view.bytes);
        transfers.push_back(
            {controls[i]->data, source, view.bytes, cudaMemcpyDeviceToHost});
      }
      if (port.host && port.control) {
        addresses[i] = controls[i]->data;
      } else if (port.host || view.bytes == 0 || view.data % 256 != 0) {
        if (!staging[i]) {
          staging[i] = std::make_unique<Buffer>(model.device, port.host);
        }
        staging[i]->reserve(view.bytes);
        addresses[i] = staging[i]->data;
        transfers.push_back(
            {addresses[i], source, view.bytes,
             port.host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice});
      } else {
        addresses[i] = source;
      }
    }
  }

  void refresh(cudaStream_t queue) const {
    for (const auto& transfer : transfers) {
      transfer.run(queue);
    }
  }

  void configure(std::span<const adapter::View> views) {
    const auto selected = model.select_profile(views, controls);
    if (selected != profile) {
      require(context->setOptimizationProfileAsync(selected, stream.value),
              "TensorRT profile selection failed");
      profile = selected;
    }
    for (size_t i = 0; i < views.size(); ++i) {
      const auto* name = model.inputs[i].name.c_str();
      invalid(context->setInputShape(
                  name, dimensions(views[i].port.shape, views[i].port.rank)),
              "engine rejected input dimensions");
      // Metadata-only probing has no input storage to bind.
      if (addresses[i]) {
        require(context->setInputTensorAddress(name, addresses[i]),
                "TensorRT input binding failed");
      }
    }
    invalid(context->inferShapes(0, nullptr) == 0, "shape inference failed");
  }

  Model& model;
  Stream& stream;
  std::vector<std::unique_ptr<Buffer>> staging, controls;
  std::vector<void*> addresses;
  std::vector<Transfer> transfers;
  int profile = -1;
  Owned<trt::IExecutionContext> context;
};

void Model::inspect_fixed_shapes() {
  if (dynamic || std::any_of(inputs.begin(), inputs.end(), [](const Port& p) {
        return !concrete(p.declared);
      })) {
    return;
  }
  // A bundle may pin a dynamic engine to concrete inputs. Probe only shape
  // inference, without activation buffers or inference, to distinguish fixed
  // bindings from data-dependent outputs that still need an allocator.
  Stream stream(device, external_stream);
  Execution execution(*this, stream,
                      trt::ExecutionContextAllocationStrategy::kUSER_MANAGED);
  Work work(stream);
  std::vector<adapter::View> views;
  for (const auto& port : inputs) {
    views.push_back({{nullptr, 0, port.dtype.code, port.declared.d,
                      static_cast<size_t>(port.declared.nbDims)},
                     0,
                     0,
                     device});
  }
  execution.configure(views);
  for (const auto& port : outputs) {
    const auto shape = execution.context->getTensorShape(port.name.c_str());
    if (concrete(shape)) {
      port.check(shape);
    } else {
      dynamic = true;
    }
  }
  work.finish();
}

struct Output : trt::IOutputAllocator {
  Output(int device, bool host) : buffer(device, host) {}

  void* reallocateOutputAsync(const char*, void*, uint64_t size,
                              uint64_t alignment,
                              cudaStream_t) noexcept override {
    try {
      buffer.reserve(size, alignment);
      return buffer.data;
    } catch (...) {
      failure = std::current_exception();
      return nullptr;
    }
  }

  void notifyShape(const char*, const trt::Dims& value) noexcept override {
    shape = value;
  }

  Buffer buffer;
  trt::Dims shape{-1, {}};
  std::exception_ptr failure;
};

struct Outputs {
  std::vector<std::unique_ptr<Output>> values;
  std::vector<adapter::View> views;

  void describe(const Model& model) {
    views.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      auto& value = *values[i];
      if (value.failure) {
        std::rethrow_exception(value.failure);
      }
      const auto& port = model.outputs[i];
      port.check(value.shape);
      const auto size = byte_size(value.shape, port.dtype.bytes);
      require(value.buffer.data != nullptr && size <= value.buffer.capacity,
              "TensorRT output allocation is incomplete");
      views.push_back({{nullptr, 0, port.dtype.code, value.shape.d,
                        static_cast<size_t>(value.shape.nbDims)},
                       reinterpret_cast<uintptr_t>(value.buffer.data),
                       size,
                       port.host ? -1 : model.device});
    }
  }
};

// Detach output allocators and addresses before their owners can be released,
// including before the next inferShapes call can write shape outputs.
struct OutputBindings {
  explicit OutputBindings(Execution& value) : execution(value) {}

  ~OutputBindings() {
    DeviceScope scope(execution.model.device);
    for (const auto& port : execution.model.outputs) {
      execution.context->setOutputAllocator(port.name.c_str(), nullptr);
      execution.context->setTensorAddress(port.name.c_str(), nullptr);
    }
  }

  Execution& execution;
};

struct Session {
  explicit Session(Model& owner)
      : model(owner), stream(owner.device, owner.external_stream) {}

  std::unique_ptr<Call> prepare(std::span<const adapter::View> inputs,
                                std::span<const adapter::View> outputs);

  std::unique_ptr<Outputs> run_dynamic(std::span<const adapter::View> inputs) {
    if (!dynamic) {
      dynamic = std::make_unique<Execution>(model, stream);
    }
    auto output = std::make_unique<Outputs>();
    OutputBindings bindings(*dynamic);
    Work work(stream);
    dynamic->inputs(inputs);
    dynamic->refresh(stream.value);
    // Shape-control values must be ready for host-side profile selection and
    // inferShapes; the same stream orders profile changes and inference.
    cuda::check(cudaStreamSynchronize(stream.value));
    dynamic->configure(inputs);
    for (const auto& port : model.outputs) {
      auto value = std::make_unique<Output>(model.device, port.host);
      value->shape = dynamic->context->getTensorShape(port.name.c_str());
      if (concrete(value->shape)) {
        port.check(value->shape);
        value->buffer.reserve(byte_size(value->shape, port.dtype.bytes));
        require(dynamic->context->setTensorAddress(port.name.c_str(),
                                                   value->buffer.data),
                "TensorRT output binding failed");
      } else {
        require(dynamic->context->setOutputAllocator(port.name.c_str(),
                                                     value.get()),
                "TensorRT output allocator binding failed");
      }
      output->values.push_back(std::move(value));
    }
    if (model.shape_outputs) {
      invalid(dynamic->context->inferShapes(0, nullptr) == 0,
              "output shape inference failed");
    }
    require(dynamic->context->enqueueV3(stream.value),
            "TensorRT enqueueV3 failed");
    work.finish();
    output->describe(model);
    return output;
  }

  Model& model;
  Stream stream;
  std::unique_ptr<Execution> dynamic;
};

struct Call {
  Call(Session& session, std::span<const adapter::View> inputs,
       std::span<const adapter::View> outputs)
      : execution(session.model, session.stream) {
    supported(!session.model.dynamic, "this model requires dynamic execution");
    Work work(session.stream);
    execution.inputs(inputs);
    execution.configure(inputs);
    invalid(outputs.size() == session.model.outputs.size(),
            "output count mismatch");
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto& view = outputs[i];
      const auto& port = session.model.outputs[i];
      const auto shape = execution.context->getTensorShape(port.name.c_str());
      port.check(shape);
      const auto destination_shape = port.check(view, session.model.device);
      invalid(std::equal(shape.d, shape.d + shape.nbDims, destination_shape.d),
              "fixed output does not match inferred engine shape");
      auto* destination = reinterpret_cast<void*>(view.data);
      void* address = destination;
      if (port.host || view.bytes == 0 || view.data % 256 != 0) {
        auto buffer = std::make_unique<Buffer>(session.model.device, port.host);
        buffer->reserve(view.bytes);
        address = buffer->data;
        transfers.push_back(
            {destination, address, view.bytes,
             port.host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice});
        staging.push_back(std::move(buffer));
      }
      require(execution.context->setTensorAddress(port.name.c_str(), address),
              "TensorRT output binding failed");
    }
    work.finish();
  }

  void run() {
    Work work(execution.stream);
    const auto stream = execution.stream.value;
    execution.refresh(stream);
    if (execution.model.shape_outputs) {
      invalid(execution.context->inferShapes(0, nullptr) == 0,
              "output shape inference failed");
    }
    require(execution.context->enqueueV3(stream), "TensorRT enqueueV3 failed");
    for (const auto& transfer : transfers) {
      transfer.run(stream);
    }
    work.finish();
  }

  std::vector<std::unique_ptr<Buffer>> staging;
  std::vector<Transfer> transfers;
  Execution execution;
};

std::unique_ptr<Session> Model::create_session() {
  return std::make_unique<Session>(*this);
}

std::unique_ptr<Call> Session::prepare(std::span<const adapter::View> inputs,
                                       std::span<const adapter::View> outputs) {
  return std::make_unique<Call>(*this, inputs, outputs);
}

// Only lifecycle transport crosses this TensorRT-specific boundary.
template <typename F>
char* invoke(F&& function) noexcept {
  try {
    diagnostic.clear();
    function();
    return nullptr;
  } catch (const std::exception& error) {
    auto* message = ::strdup(error.what());
    if (!message) {
      std::abort();
    }
    return message;
  } catch (...) {
    auto* message = ::strdup("unknown TensorRT failure");
    if (!message) {
      std::abort();
    }
    return message;
  }
}

const adapter::Api api{
    [](const char* path, int32_t device, void* stream, const adapter::Port* inputs,
       size_t input_count, const adapter::Port* outputs, size_t output_count,
       void** value) noexcept {
      return invoke([&] {
        *value = new Model(path, device, static_cast<cudaStream_t>(stream),
                           {inputs, input_count}, {outputs, output_count});
      });
    },
    [](void* value) noexcept { delete static_cast<Model*>(value); },
    [](const void* value) noexcept {
      return static_cast<const Model*>(value)->requires_dynamic();
    },
    [](void* model, void** value) noexcept {
      return invoke([&] {
        *value = static_cast<Model*>(model)->create_session().release();
      });
    },
    [](void* value) noexcept { delete static_cast<Session*>(value); },
    [](void* session, const adapter::View* inputs, size_t input_count,
       const adapter::View* outputs, size_t output_count,
       void** value) noexcept {
      return invoke([&] {
        *value = static_cast<Session*>(session)
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
        *value = static_cast<Session*>(session)
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
}  // namespace leapp_tensorrt

extern "C" __attribute__((visibility("default")))
const leapp_native::bridge::Api*
leapp_native_tensorrt() noexcept {
  return &leapp_tensorrt::api;
}
