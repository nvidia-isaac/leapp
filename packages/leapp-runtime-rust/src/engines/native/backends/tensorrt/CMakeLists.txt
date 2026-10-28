# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(leapp_native_tensorrt LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The TensorRT adapter currently requires Linux")
endif()
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT SDK root")
set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA SDK root")
find_path(TENSORRT_INCLUDE_DIR NvInferRuntime.h HINTS "${TENSORRT_ROOT}"
  PATH_SUFFIXES include "include/${CMAKE_LIBRARY_ARCHITECTURE}" REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer HINTS "${TENSORRT_ROOT}"
  PATH_SUFFIXES lib lib64 "lib/${CMAKE_LIBRARY_ARCHITECTURE}" REQUIRED)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin HINTS "${TENSORRT_ROOT}"
  PATH_SUFFIXES lib lib64 "lib/${CMAKE_LIBRARY_ARCHITECTURE}" REQUIRED)
find_library(CUDA_LIBRARY cudart PATHS "${CUDA_ROOT}/lib64" NO_DEFAULT_PATH REQUIRED)

add_library(leapp_tensorrt SHARED backend.cc)
target_compile_features(leapp_tensorrt PRIVATE cxx_std_20)
target_include_directories(leapp_tensorrt PRIVATE "${TENSORRT_INCLUDE_DIR}" "${CUDA_ROOT}/include")
# Older engines may store 32-bit shape bounds; their reader is deprecated but
# still required alongside the newer 64-bit shape-bound API.
target_compile_options(leapp_tensorrt PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations)
target_link_libraries(leapp_tensorrt PRIVATE "${TENSORRT_LIBRARY}" "${TENSORRT_PLUGIN_LIBRARY}" "${CUDA_LIBRARY}")
target_link_options(leapp_tensorrt PRIVATE "LINKER:--no-undefined")
set_target_properties(leapp_tensorrt PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
  BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN;$ORIGIN/lib")
install(TARGETS leapp_tensorrt LIBRARY DESTINATION tensorrt)
