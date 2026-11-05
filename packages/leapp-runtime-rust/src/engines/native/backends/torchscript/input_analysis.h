// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/script.h>

#include "../torch/model.h"

namespace leapp_torch::native {

InputPlan analyze_inputs(const torch::jit::Module& module);

}  // namespace leapp_torch::native
