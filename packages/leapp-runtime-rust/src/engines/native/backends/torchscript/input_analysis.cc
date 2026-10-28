// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "input_analysis.h"

#include <exception>

#include "torch/csrc/jit/ir/alias_analysis.h"
#include "torch/csrc/jit/passes/inliner.h"

namespace leapp_torch::native {
namespace {

bool RequiresOwnedInputs(const torch::jit::Block* block,
                         const torch::jit::AliasDb& aliases,
                         bool& background_work) {
  bool owned = false;
  for (const auto* node : block->nodes()) {
    // Subgraphs such as fork bodies can retain inputs beyond this invocation.
    for (const auto name : node->attributeNames()) {
      if (node->kindOf(name) == torch::jit::AttributeKind::g ||
          node->kindOf(name) == torch::jit::AttributeKind::gs) {
        background_work = true;
        owned = true;
      }
    }
    if (node->kind() == torch::jit::prim::SetAttr) {
      owned = true;
    }
    if (node->kind() == torch::jit::prim::GetAttr) {
      for (const auto* value : node->outputs()) {
        if (aliases.hasWriters(value)) {
          owned = true;
        }
      }
    }
    for (const auto* child : node->blocks()) {
      owned = RequiresOwnedInputs(child, aliases, background_work) || owned;
    }
  }
  return owned;
}

}  // namespace

InputPlan analyze_inputs(const torch::jit::Module& module) {
  const auto method = module.find_method("forward");
  TORCH_CHECK(method, "TorchScript model is missing forward");
  const auto& arguments = method->function().getSchema().arguments();
  TORCH_CHECK(!arguments.empty() &&
                  arguments.front().type()->kind() == c10::TypeKind::ClassType,
              "invalid TorchScript forward signature");
  for (size_t i = 1; i < arguments.size(); ++i) {
    TORCH_CHECK(arguments[i].type()->kind() == c10::TypeKind::TensorType &&
                    !arguments[i].kwarg_only(),
                "TorchScript forward requires positional tensor arguments");
  }
  InputPlan plan{
      std::vector<InputAccess>(arguments.size() - 1, InputAccess::Owned)};
  auto& access = plan.access;
  // Analyze a copy; do not change the graph that the executor will optimize.
  try {
    auto graph = method->graph()->copy();
    torch::jit::Inline(*graph);
    // A saved input must own storage even if forward does not mutate it now.
    torch::jit::AliasDb aliases(graph);
    if (RequiresOwnedInputs(graph->block(), aliases, plan.background_work)) {
      return plan;
    }
    for (size_t i = 0; i < access.size(); ++i) {
      access[i] = aliases.hasWriters(graph->inputs().at(i + 1))
                      ? InputAccess::Scratch
                      : InputAccess::Borrow;
    }
  } catch (const c10::Error&) {
    // Unsupported analysis must never turn into permission to borrow.
    access.assign(arguments.size() - 1, InputAccess::Owned);
    plan.background_work = true;
  }
  return plan;
}

}  // namespace leapp_torch::native
