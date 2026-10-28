# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(leapp_native_nativert LANGUAGES CXX)
add_subdirectory(../torch torch)
set(PYTORCH_SOURCE "$ENV{PYTORCH_SOURCE}" CACHE PATH "Matching PyTorch v2.12.1 source with submodules")
set(NATIVERT_LIBRARY "$ENV{NATIVERT_LIBRARY}" CACHE FILEPATH "NativeRT implementation built for this LibTorch distribution")
foreach(required IN ITEMS
    "${PYTORCH_SOURCE}/version.txt"
    "${PYTORCH_SOURCE}/torch/nativert/executor/Executor.h"
    "${PYTORCH_SOURCE}/third_party/nlohmann/include/nlohmann/json.hpp"
    "${PYTORCH_SOURCE}/third_party/concurrentqueue/moodycamel/concurrentqueue.h"
    "${TORCH_ROOT}/include/torch/headeronly/version.h"
    "${NATIVERT_LIBRARY}")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "Missing NativeRT dependency: ${required}")
  endif()
endforeach()
file(READ "${PYTORCH_SOURCE}/version.txt" source_version)
string(STRIP "${source_version}" source_version)
file(READ "${TORCH_ROOT}/include/torch/headeronly/version.h" torch_version)
if(NOT source_version MATCHES "^2\\.12\\.1(a0)?$" OR
   NOT torch_version MATCHES "#define TORCH_VERSION_MAJOR 2\n" OR
   NOT torch_version MATCHES "#define TORCH_VERSION_MINOR 12\n" OR
   NOT torch_version MATCHES "#define TORCH_VERSION_PATCH 1\n")
  message(FATAL_ERROR "Use PyTorch v2.12.1 source and LibTorch 2.12.1 together")
endif()
add_library(leapp_nativert_backend SHARED model.cc)
# The dependency already uses libleapp_nativert.so; keep the adapter distinct.
set_target_properties(leapp_nativert_backend PROPERTIES
  OUTPUT_NAME leapp_nativert_backend
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
  BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN;$ORIGIN/lib")
target_link_libraries(leapp_nativert_backend PRIVATE
  leapp_native_torch "${NATIVERT_LIBRARY}" ${CMAKE_DL_LIBS})
target_compile_definitions(leapp_nativert_backend PRIVATE
  FMT_HEADER_ONLY LEAPP_TORCH_ENTRY=leapp_native_nativert_v2)
target_include_directories(leapp_nativert_backend PRIVATE
  "${PYTORCH_SOURCE}"
  "${PYTORCH_SOURCE}/third_party/nlohmann/include"
  "${PYTORCH_SOURCE}/third_party/concurrentqueue")
install(TARGETS leapp_nativert_backend LIBRARY DESTINATION nativert)
