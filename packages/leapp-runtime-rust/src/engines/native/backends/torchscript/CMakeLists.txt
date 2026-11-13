# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(leapp_native_torchscript LANGUAGES CXX)
add_subdirectory(../torch torch)
add_library(leapp_torchscript SHARED model.cc input_analysis.cc)
target_link_libraries(leapp_torchscript PRIVATE leapp_native_torch)
target_compile_definitions(leapp_torchscript PRIVATE LEAPP_TORCH_ENTRY=leapp_native_torchscript)
set_target_properties(leapp_torchscript PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
  BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN;$ORIGIN/lib")
install(TARGETS leapp_torchscript LIBRARY DESTINATION torchscript)
