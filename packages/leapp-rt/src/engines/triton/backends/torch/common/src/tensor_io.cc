// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/tensor_io.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "ATen/Functions.h"
#include "c10/cuda/CUDAException.h"
#include "src/backend_utils.h"

namespace leapp::internal {

at::ScalarType TorchType(TRITONSERVER_DataType type) {
  switch (type) {
    case TRITONSERVER_TYPE_BOOL:
      return at::kBool;
    case TRITONSERVER_TYPE_UINT8:
      return at::kByte;
    case TRITONSERVER_TYPE_UINT16:
      return at::ScalarType::UInt16;
    case TRITONSERVER_TYPE_UINT32:
      return at::ScalarType::UInt32;
    case TRITONSERVER_TYPE_UINT64:
      return at::ScalarType::UInt64;
    case TRITONSERVER_TYPE_INT8:
      return at::kChar;
    case TRITONSERVER_TYPE_INT16:
      return at::kShort;
    case TRITONSERVER_TYPE_INT32:
      return at::kInt;
    case TRITONSERVER_TYPE_INT64:
      return at::kLong;
    case TRITONSERVER_TYPE_FP16:
      return at::kHalf;
    case TRITONSERVER_TYPE_BF16:
      return at::kBFloat16;
    case TRITONSERVER_TYPE_FP32:
      return at::kFloat;
    case TRITONSERVER_TYPE_FP64:
      return at::kDouble;
    default:
      return at::ScalarType::Undefined;
  }
}

bool MatchesShape(std::span<const int64_t> expected, at::IntArrayRef actual) {
  return static_cast<size_t>(expected.size()) == actual.size() &&
         std::equal(expected.begin(), expected.end(), actual.begin(),
                    [](int64_t declared, int64_t dimension) {
                      return declared == -1 || declared == dimension;
                    });
}

Status PrepareInput(at::IntArrayRef shape, at::ScalarType dtype,
                    std::span<const InputBuffer> buffers, InputAccess access,
                    c10::Device device, at::Tensor* tensor) {
  uint64_t bytes = c10::elementSize(dtype);
  for (int64_t dimension : shape) {
    if (dimension < 0 ||
        (dimension != 0 &&
         bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                     static_cast<uint64_t>(dimension))) {
      return {ErrorCode::kInvalidArgument, "backend", "invalid tensor size"};
    }
    bytes *= static_cast<uint64_t>(dimension);
  }
  uint64_t total = 0;
  for (const auto& buffer : buffers) {
    if (buffer.bytes > bytes - total ||
        (buffer.bytes != 0 && buffer.data == nullptr)) {
      return {ErrorCode::kInvalidArgument, "backend", "invalid input buffer"};
    }
    total += buffer.bytes;
  }
  if (total != bytes) {
    return {ErrorCode::kInvalidArgument, "backend", "input byte size mismatch"};
  }
  const auto options = at::TensorOptions().dtype(dtype).device(device);
  if (bytes != 0 && access == InputAccess::kBorrowReadOnly &&
      buffers.size() == 1 && buffers.front().device == device) {
    // The request keeps borrowed storage alive through stream completion.
    *tensor =
        at::from_blob(const_cast<void*>(buffers.front().data), shape, options);
    return {};
  }
  // Gathering, transfers, and mutation protection share one owned allocation.
  *tensor = at::empty(shape, options);
  auto storage = tensor->view(at::kByte).reshape({-1});
  int64_t offset = 0;
  for (const auto& buffer : buffers) {
    if (buffer.bytes != 0) {
      auto chunk = at::from_blob(
          const_cast<void*>(buffer.data), {static_cast<int64_t>(buffer.bytes)},
          at::TensorOptions().dtype(at::kByte).device(buffer.device));
      storage.narrow(0, offset, static_cast<int64_t>(buffer.bytes))
          .copy_(chunk);
    }
    offset += static_cast<int64_t>(buffer.bytes);
  }
  return {};
}

Status ReadInputs(TRITONBACKEND_Request* request,
                  std::span<const ModelPort> ports,
                  std::span<const InputAccess> access, c10::Device device,
                  std::vector<c10::IValue>* inputs) {
  if (ports.size() != access.size()) {
    return {ErrorCode::kInternal, "backend", "input plan size mismatch"};
  }
  for (size_t index = 0; index < ports.size(); ++index) {
    const auto& port = ports[index];
    TRITONBACKEND_Input* input = nullptr;
    auto status = TakeBackendError(
        TRITONBACKEND_RequestInput(request, port.name.c_str(), &input));
    if (!status.ok()) {
      return status;
    }
    TRITONSERVER_DataType dtype{};
    const int64_t* shape = nullptr;
    uint32_t rank = 0;
    uint64_t bytes = 0;
    uint32_t chunks = 0;
    status = TakeBackendError(TRITONBACKEND_InputProperties(
        input, nullptr, &dtype, &shape, &rank, &bytes, &chunks));
    if (!status.ok()) {
      return status;
    }
    const at::IntArrayRef dimensions(shape, rank);
    if (dtype != port.dtype || TorchType(dtype) == at::ScalarType::Undefined ||
        !MatchesShape(port.shape, dimensions)) {
      return {ErrorCode::kInvalidArgument, "backend",
              port.name + ": input dtype or shape does not match"};
    }
    std::vector<InputBuffer> buffers;
    uint64_t total = 0;
    for (uint32_t i = 0; i < chunks; ++i) {
      const void* data = nullptr;
      uint64_t size = 0;
      TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t memory_id = 0;
      status = TakeBackendError(TRITONBACKEND_InputBuffer(
          input, i, &data, &size, &memory_type, &memory_id));
      if (!status.ok()) {
        return status;
      }
      if (size > bytes - total) {
        return {ErrorCode::kInvalidArgument, "backend", "invalid input size"};
      }
      c10::Device source(c10::kCPU);
      status = BufferDevice(memory_type, memory_id, &source);
      if (!status.ok()) {
        return status;
      }
      buffers.push_back({data, size, source});
      total += size;
    }
    if (total != bytes) {
      return {ErrorCode::kInvalidArgument, "backend",
              "incomplete input buffers"};
    }
    // Publish the owner before any asynchronous copies can be queued.
    inputs->emplace_back(at::Tensor());
    auto& tensor = inputs->back().toTensor();
    status = PrepareInput(dimensions, TorchType(dtype), buffers, access[index],
                          device, &tensor);
    if (!status.ok()) {
      return status;
    }
  }
  return {};
}

Status WriteOutputs(TRITONBACKEND_Request* request,
                    std::span<const ModelPort> ports,
                    std::span<const c10::IValue> outputs,
                    TRITONBACKEND_Response* response,
                    const std::optional<c10::cuda::CUDAStream>& stream) {
  if (outputs.size() != static_cast<size_t>(ports.size())) {
    return {ErrorCode::kInternal, "backend",
            "model output count does not match"};
  }
  uint32_t count = 0;
  auto status =
      TakeBackendError(TRITONBACKEND_RequestOutputCount(request, &count));
  if (!status.ok()) {
    return status;
  }
  for (uint32_t i = 0; i < count; ++i) {
    const char* name = nullptr;
    status =
        TakeBackendError(TRITONBACKEND_RequestOutputName(request, i, &name));
    if (!status.ok()) {
      return status;
    }
    const auto port =
        std::find_if(ports.begin(), ports.end(),
                     [name](const auto& p) { return p.name == name; });
    if (port == ports.end()) {
      return {ErrorCode::kInvalidArgument, "backend",
              "unknown requested output"};
    }
    const auto& value = outputs[static_cast<size_t>(port - ports.begin())];
    const auto dtype = port->dtype;
    if (!value.isTensor() ||
        value.toTensor().scalar_type() != TorchType(dtype) ||
        !MatchesShape(port->shape, value.toTensor().sizes())) {
      return {ErrorCode::kInternal, "backend",
              port->name + ": output dtype or shape does not match"};
    }
    const auto& tensor = value.toTensor();
    TRITONBACKEND_Output* output = nullptr;
    status = TakeBackendError(TRITONBACKEND_ResponseOutput(
        response, &output, name, dtype, tensor.sizes().data(),
        static_cast<uint32_t>(tensor.dim())));
    if (!status.ok()) {
      return status;
    }
    void* data = nullptr;
    auto memory_type =
        tensor.is_cuda() ? TRITONSERVER_MEMORY_GPU : TRITONSERVER_MEMORY_CPU;
    int64_t memory_id = tensor.is_cuda() ? tensor.get_device() : 0;
    status = TakeBackendError(TRITONBACKEND_OutputBuffer(
        output, &data, tensor.nbytes(), &memory_type, &memory_id));
    if (!status.ok()) {
      return status;
    }
    c10::Device destination(c10::kCPU);
    status = BufferDevice(memory_type, memory_id, &destination);
    if (!status.ok()) {
      return status;
    }
    if (tensor.nbytes() != 0) {
      if (data == nullptr) {
        return {ErrorCode::kInternal, "backend", "null output buffer"};
      }
      // Plain CUDA storage needs no temporary Tensor wrapper. Other layouts,
      // view bits, and cross-device transfers retain Torch's copy semantics.
      if (stream && tensor.is_cuda() &&
          tensor.get_device() == stream->device_index() &&
          (destination.is_cpu() || destination == tensor.device()) &&
          tensor.is_contiguous() && !tensor.is_neg() && !tensor.is_conj()) {
        C10_CUDA_CHECK(cudaMemcpyAsync(data, tensor.data_ptr(), tensor.nbytes(),
                                       destination.is_cpu()
                                           ? cudaMemcpyDeviceToHost
                                           : cudaMemcpyDeviceToDevice,
                                       stream->stream()));
      } else {
        auto target = at::from_blob(data, tensor.sizes(),
                                    tensor.options().device(destination));
        target.copy_(tensor);
      }
    }
  }
  return {};
}

}  // namespace leapp::internal
