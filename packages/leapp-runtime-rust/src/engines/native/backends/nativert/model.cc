// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include "../torch/model.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/InferenceMode.h>
#include <caffe2/serialize/inline_container.h>
#include <fmt/format.h>
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/nativert/executor/Executor.h>
#include <torch/nativert/graph/GraphPasses.h>
#include <torch/nativert/graph/Serialization.h>
#include <torch/nativert/kernels/KernelHandlerRegistry.h>

#include <mutex>
#include <nlohmann/json.hpp>

namespace leapp_torch::native {

namespace {

namespace nr = torch::nativert;
namespace archive = torch::_export::archive_spec;
using Reader = caffe2::serialize::PyTorchStreamReader;
using Json = nlohmann::json;

Json read_json(Reader& reader, const std::string& path) {
  TORCH_CHECK(reader.hasRecord(path), "PT2 archive is missing ", path);
  const auto [data, size] = reader.getRecord(path);
  const auto* begin = static_cast<const char*>(data.get());
  return Json::parse(begin, begin + size);
}

std::unordered_map<std::string, std::string> payloads(Reader& reader,
                                                      std::string_view format) {
  const auto json =
      read_json(reader, fmt::format(fmt::runtime(format), "model"));
  const auto config = json.get<torch::_export::PayloadConfig>();
  std::unordered_map<std::string, std::string> result;
  for (const auto& [name, value] : config.get_config()) {
    result.emplace(name, value.get_path_name());
  }
  return result;
}

void check_port(const nr::Graph& graph, const std::string& name,
                const Port& port) {
  const auto it = graph.tensorValuesMeta().find(name);
  TORCH_CHECK(it != graph.tensorValuesMeta().end(),
              "PT2 ports must be tensors: ", name);
  const auto& meta = it->second;
  TORCH_CHECK(meta.dtype() == scalar_type(port.dtype) &&
                  meta.dim() == static_cast<int64_t>(port.shape.size()),
              "PT2 port dtype or rank mismatch: ", name);
  if (!meta.hasSymbolicShape()) {
    for (size_t i = 0; i < port.shape.size(); ++i) {
      TORCH_CHECK(port.shape[i] == -1 || port.shape[i] == meta.sizes()[i],
                  "PT2 port shape mismatch: ", name);
    }
  }
}

InputPlan analyze(const nr::Graph& graph) {
  InputAccess access = InputAccess::Borrow;
  bool background = false;
  for (const auto& node : graph.nodes()) {
    const auto target = node.target();
    if (target.starts_with("torch.ops.aten.")) {
      try {
        const auto op = nr::getOperatorForTarget(target, &node);
        const auto& schema = op.schema();
        if (schema.name() == "aten::set_") {
          access = InputAccess::Owned;
        } else if (schema.is_mutable()) {
          access = std::max(access, InputAccess::Scratch);
        }
      } catch (const c10::Error&) {
        access = InputAccess::Owned;
        background = true;
      }
    } else if (target != "prim.Input" && target != "prim.Output" &&
               target != "prim.ListPack" && target != "prim.ListUnpack" &&
               !target.starts_with("_operator.")) {
      // Custom operators and subgraphs may retain tensors or launch other
      // streams.
      access = InputAccess::Owned;
      background = true;
    }
  }
  // Conservatively protect every input if any ATen argument can be written:
  // views, packed lists and repeated inputs may alias the written argument.
  return {std::vector<InputAccess>(graph.userInputs().size(), access),
          background};
}

class Pt2Model final : public Model {
 public:
  Pt2Model(const std::string& path, at::Device device,
           std::span<const Port> inputs, std::span<const Port> outputs)
      : reader_(std::make_shared<Reader>(path)) {
    nr::register_kernel_handlers();
    auto json = read_json(*reader_, "models/model.json");
    auto& specs = json.at("graph_module").at("signature").at("output_specs");
    // NativeRT's flat runner returns only user outputs. Include mutation
    // results privately so we can commit buffer updates without publishing
    // extra ports.
    for (size_t i = 0; i < specs.size(); ++i) {
      auto& spec = specs[i];
      if (spec.contains("user_output")) {
        user_outputs_.push_back(i);
      } else if (spec.contains("buffer_mutation")) {
        const auto mutation = spec.at("buffer_mutation");
        updates_.push_back(
            {i, mutation.at("buffer_name").get<std::string>(), {}});
        spec = Json{
            {"user_output", {{"arg", {{"as_tensor", mutation.at("arg")}}}}}};
      } else if (spec.contains("user_input_mutation")) {
        const auto arg = spec.at("user_input_mutation").at("arg");
        spec = Json{{"user_output", {{"arg", {{"as_tensor", arg}}}}}};
      } else {
        TORCH_CHECK(false, "unsupported PT2 output signature");
      }
    }
    const auto program = json.get<torch::_export::ExportedProgram>();
    graph_ = nr::jsonToGraph(program.get_graph_module());
    graph_->applyDevicePlacement(nr::Placement(device));
    nr::selectScalarOverload(graph_.get());
    TORCH_CHECK(graph_->userInputs().size() == inputs.size() &&
                    user_outputs_.size() == outputs.size(),
                "PT2 port count mismatch");
    for (size_t i = 0; i < inputs.size(); ++i) {
      check_port(*graph_, std::string(graph_->userInputs()[i]->name()),
                 inputs[i]);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto& value = graph_->userOutputs().at(user_outputs_[i]);
      TORCH_CHECK(std::holds_alternative<nr::Value*>(value),
                  "PT2 outputs must be tensors");
      check_port(*graph_, std::string(std::get<nr::Value*>(value)->name()),
                 outputs[i]);
    }
    plan = analyze(*graph_);
    weights_ = std::make_shared<nr::Weights>(
        graph_.get(), reader_,
        payloads(*reader_, archive::WEIGHTS_CONFIG_FILENAME_FORMAT),
        archive::WEIGHTS_DIR,
        payloads(*reader_, archive::CONSTANTS_CONFIG_FILENAME_FORMAT),
        archive::CONSTANTS_DIR);
    weights_->validateAllWeightsLoaded();
    for (auto& update : updates_) {
      update.scratch = at::empty_like(weights_->at(update.name));
    }
    nr::ExecutorConfig config;
    config.modelName = "model";
    config.validateInputs = true;
    config.enableStaticCPUKernels = false;
    config.maxNumConcurrentThreads = 1;
    config.maxParallelOps = 1;
    config.doExecutionFrameCleanup = false;
    // Clear transients on both success and failure before reusing the frame.
    config.tryFreeUnmanagedValuesAfterUse = false;
    executor_ =
        std::make_unique<nr::Executor>(config, graph_, weights_, reader_);
  }

  std::vector<at::Tensor> forward(
      const std::vector<c10::IValue>& args) override {
    c10::InferenceMode inference;
    auto values = executor_->execute(args);
    // Snapshot all updates before copying: two state outputs may alias buffers
    // that another update overwrites. Input mutation outputs remain private.
    for (auto& update : updates_) {
      TORCH_CHECK(values.at(update.output).isTensor(),
                  "PT2 buffer update must be a tensor");
      const auto& value = values[update.output].toTensor();
      weights_->validateValue(update.name, value);
      update.scratch.copy_(value);
    }
    for (const auto& update : updates_) {
      weights_->updateValue(update.name, update.scratch);
    }
    std::vector<at::Tensor> result;
    for (auto index : user_outputs_) {
      TORCH_CHECK(values.at(index).isTensor(), "PT2 outputs must be tensors");
      result.push_back(values[index].toTensor());
    }
    return result;
  }

 private:
  std::shared_ptr<Reader> reader_;
  std::shared_ptr<nr::Graph> graph_;
  std::shared_ptr<nr::Weights> weights_;
  std::unique_ptr<nr::Executor> executor_;
  std::vector<size_t> user_outputs_;

  struct BufferUpdate {
    size_t output;
    std::string name;
    at::Tensor scratch;
  };

  std::vector<BufferUpdate> updates_;
};

}  // namespace

std::unique_ptr<Model> load_model(const std::string& path, at::Device device,
                                  std::span<const Port> inputs,
                                  std::span<const Port> outputs) {
  // This SDK assigns Weights versions through a non-atomic global counter.
  // Serialize construction only; independent sessions execute without this
  // lock.
  static std::mutex loading;
  std::lock_guard lock(loading);
  return std::make_unique<Pt2Model>(path, device, inputs, outputs);
}

}  // namespace leapp_torch::native
