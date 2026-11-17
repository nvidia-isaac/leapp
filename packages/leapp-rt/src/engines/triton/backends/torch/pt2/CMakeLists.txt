# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(leapp_nativert_backend LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The NativeRT backend requires Linux")
endif()
add_subdirectory(../common common)
set(NATIVERT_LIBRARY "" CACHE FILEPATH
  "NativeRT implementation built for this LibTorch distribution")
if(NOT EXISTS "${TORCH_ROOT}/include/torch/csrc/api/include/torch/nativert/ModelRunnerHandle.h")
  message(FATAL_ERROR "Supply TORCH_ROOT with NativeRT headers")
endif()
if(NOT EXISTS "${NATIVERT_LIBRARY}")
  message(FATAL_ERROR
    "Supply NATIVERT_LIBRARY; headers alone are insufficient. "
    "See ../nativert for the pinned fallback build.")
endif()

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES leapp_torch "${NATIVERT_LIBRARY}")
unset(LEAPP_NATIVERT_LINKS CACHE)
check_cxx_source_compiles([=[
  #include <torch/nativert/ModelRunnerHandle.h>
  int main() {
    torch::nativert::ModelRunnerHandle model("probe.pt2", "model");
    model.runWithFlatInputsAndOutputs({});
  }
]=] LEAPP_NATIVERT_LINKS)
unset(CMAKE_REQUIRED_LIBRARIES)
if(NOT LEAPP_NATIVERT_LINKS)
  message(FATAL_ERROR "NativeRT failed its link check; verify LibTorch version and ABI")
endif()

add_library(triton_leapp_pt2 SHARED
  src/backend.cc
  src/native_rt_instance.cc)
target_include_directories(triton_leapp_pt2 PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(triton_leapp_pt2 PRIVATE
  leapp_backend_common "${NATIVERT_LIBRARY}" ${CMAKE_DL_LIBS})
target_compile_options(triton_leapp_pt2 PRIVATE -Wall -Wextra -Wpedantic)
target_link_options(triton_leapp_pt2 PRIVATE "LINKER:--no-undefined")
set_target_properties(triton_leapp_pt2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/backends/leapp_pt2")
install(TARGETS triton_leapp_pt2 LIBRARY DESTINATION backends/leapp_pt2)
