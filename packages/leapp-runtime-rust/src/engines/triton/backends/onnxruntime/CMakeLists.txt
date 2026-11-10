# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.31.8)
project(leapp_onnxruntime_backend LANGUAGES C CXX)
# Triton's common utilities still configure an older GoogleTest dependency.
set(CMAKE_POLICY_VERSION_MINIMUM 3.10)

include(FetchContent)
find_package(Git REQUIRED)
set(TRITON_ROOT "/opt/tritonserver" CACHE PATH "Triton installation")
find_path(ORT_INCLUDE_DIR onnxruntime_c_api.h HINTS "$ENV{ORT_INCLUDE_DIR}" REQUIRED)
set(TRITON_ONNXRUNTIME_INCLUDE_PATHS "${ORT_INCLUDE_DIR}")
set(TRITON_ONNXRUNTIME_LIB_PATHS "${TRITON_ROOT}/backends/onnxruntime")
set(TRITON_ENABLE_ONNXRUNTIME_TENSORRT ON)
set(TRITON_ENABLE_ONNXRUNTIME_OPENVINO OFF)

# Match the supported Triton r26.09 release, including its backend utilities.
set(TRITON_COMMON_REPO_TAG cc0e85d0cb0ec1e8abe1f57424eeb81ff1e1e671
  CACHE STRING "Triton common revision")
set(TRITON_CORE_REPO_TAG b9ef1524379c71466ff66ea0fda4730ddd9eec80
  CACHE STRING "Triton core revision")
set(TRITON_BACKEND_REPO_TAG cb3b36a77377d3d0492697eccd882508ac8c2198
  CACHE STRING "Triton backend revision")
# Fetch full histories so pinned commits remain available as release branches move.
foreach(component IN ITEMS COMMON CORE BACKEND)
  string(TOLOWER "${component}" repository)
  FetchContent_Declare(repo-${repository}
    GIT_REPOSITORY https://github.com/triton-inference-server/${repository}.git
    GIT_TAG "${TRITON_${component}_REPO_TAG}")
endforeach()
FetchContent_Declare(leapp_triton_onnxruntime
  EXCLUDE_FROM_ALL TRUE
  GIT_REPOSITORY https://github.com/triton-inference-server/onnxruntime_backend.git
  GIT_TAG 1d2778ae7c052cbd80eedfed4ab30ae304bd030f
  PATCH_COMMAND "${GIT_EXECUTABLE}" apply "${CMAKE_CURRENT_SOURCE_DIR}/tensorrt-options.patch")
FetchContent_MakeAvailable(leapp_triton_onnxruntime)
set_target_properties(triton-onnxruntime-backend PROPERTIES
  LINK_FLAGS "-Wl,--version-script=${leapp_triton_onnxruntime_BINARY_DIR}/libtriton_onnxruntime.ldscript")
add_custom_target(leapp_onnxruntime ALL DEPENDS triton-onnxruntime-backend)
install(FILES "$<TARGET_FILE:triton-onnxruntime-backend>"
  DESTINATION backends/onnxruntime)

foreach(component IN ITEMS leapp_triton_onnxruntime repo-common repo-core repo-backend)
  FetchContent_GetProperties(${component} SOURCE_DIR source)
  install(FILES "${source}/LICENSE"
    DESTINATION "share/licenses/leapp-backend-onnx/${component}")
endforeach()
