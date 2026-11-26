// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/input_analysis.h"

#include <exception>

#include "torch/csrc/jit/ir/alias_analysis.h"
#include "torch/csrc/jit/passes/inliner.h"

namespace leapp::internal {
namespace {

bool RequiresOwnedInputs(const torch::jit::Block* block,
                         const torch::jit::AliasDb& aliases) {
  for (const auto* node : block->nodes()) {
    // Subgraphs such as fork bodies can retain inputs beyond this invocation.
    for (const auto name : node->attributeNames()) {
      if (node->kindOf(name) == torch::jit::AttributeKind::g ||
          node->kindOf(name) == torch::jit::AttributeKind::gs) {
        return true;
      }
    }
    if (node->kind() == torch::jit::prim::SetAttr) {
      return true;
    }
    if (node->kind() == torch::jit::prim::GetAttr) {
      for (const auto* value : node->outputs()) {
        if (aliases.hasWriters(value)) {
          return true;
        }
      }
    }
    for (const auto* child : node->blocks()) {
      if (RequiresOwnedInputs(child, aliases)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

Status AnalyzeInputs(const torch::jit::Module& module, InputPlan* plan) {
  const auto method = module.find_method("forward");
  if (!method) {
    return {ErrorCode::kUnsupported, "torchscript", "missing forward method"};
  }
  const auto& arguments = method->function().getSchema().arguments();
  if (arguments.empty() ||
      arguments.front().type()->kind() != c10::TypeKind::ClassType) {
    return {ErrorCode::kUnsupported, "torchscript",
            "invalid forward signature"};
  }
  for (size_t i = 1; i < arguments.size(); ++i) {
    if (arguments[i].type()->kind() != c10::TypeKind::TensorType ||
        arguments[i].kwarg_only()) {
      return {ErrorCode::kUnsupported, "torchscript",
              "forward requires positional tensor arguments"};
    }
  }
  plan->access.assign(arguments.size() - 1, InputAccess::kRequireOwnedStorage);
  // Analyze a copy; do not change the graph that the executor will optimize.
  try {
    auto graph = method->graph()->copy();
    torch::jit::Inline(*graph);
    // A saved input must own storage even if forward does not mutate it now.
    torch::jit::AliasDb aliases(graph);
    if (RequiresOwnedInputs(graph->block(), aliases)) {
      return {};
    }
    for (size_t i = 0; i < plan->access.size(); ++i) {
      if (!aliases.hasWriters(graph->inputs().at(i + 1))) {
        plan->access[i] = InputAccess::kBorrowReadOnly;
      }
    }
  } catch (const c10::Error&) {
    // Unsupported analysis must never turn into permission to borrow.
    plan->access.assign(arguments.size() - 1,
                        InputAccess::kRequireOwnedStorage);
  }
  return {};
}

}  // namespace leapp::internal
