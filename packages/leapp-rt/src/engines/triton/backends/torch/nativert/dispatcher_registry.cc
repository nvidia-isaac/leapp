// Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "c10/util/Registry.h"
#include "torch/nativert/executor/OpKernel.h"
#include "torch/nativert/graph/Graph.h"
#include "torch/nativert/kernels/KernelRegistry.h"

namespace torch::nativert {

// NativeRT's KernelFactory falls back to C10Kernel when no static CPU kernel
// is registered. This avoids linking optional private ATen implementations.
C10_DEFINE_REGISTRY(StaticallyDispatchedCPUKernelRegistry, OpKernel,
                    const Node*)

}  // namespace torch::nativert
