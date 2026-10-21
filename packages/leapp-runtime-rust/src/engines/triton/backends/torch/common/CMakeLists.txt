# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(TRITON_ROOT "/opt/tritonserver" CACHE PATH "Triton installation")
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json 3.11 REQUIRED CONFIG)
find_path(TRITON_INCLUDE_DIR triton/core/tritonbackend.h
  HINTS "${TRITON_ROOT}/include" REQUIRED)
find_library(TRITON_SERVER_LIBRARY tritonserver
  HINTS "${TRITON_ROOT}/lib" REQUIRED)
add_library(triton_server SHARED IMPORTED GLOBAL)
set_target_properties(triton_server PROPERTIES
  IMPORTED_LOCATION "${TRITON_SERVER_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${TRITON_INCLUDE_DIR}")

set(TORCH_ROOT "" CACHE PATH "LibTorch or torch package with include/ and lib/")
foreach(name IN ITEMS torch torch_cpu torch_cuda c10 c10_cuda)
  find_library(LEAPP_${name}_LIBRARY NAMES ${name}
    PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
endforeach()
add_library(leapp_torch INTERFACE IMPORTED GLOBAL)
target_include_directories(leapp_torch SYSTEM INTERFACE
  "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include")
target_link_libraries(leapp_torch INTERFACE
  "${LEAPP_torch_LIBRARY}" "${LEAPP_torch_cpu_LIBRARY}"
  "${LEAPP_torch_cuda_LIBRARY}" "${LEAPP_c10_LIBRARY}" "${LEAPP_c10_cuda_LIBRARY}"
  CUDA::cudart)
# Match the C++11 ABI used by the supported LibTorch distribution.
target_compile_definitions(leapp_torch INTERFACE _GLIBCXX_USE_CXX11_ABI=1)
add_library(leapp_backend_common STATIC
  src/backend_utils.cc src/model_config.cc src/tensor_io.cc)
set_target_properties(leapp_backend_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(leapp_backend_common PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(leapp_backend_common PUBLIC
  leapp_torch triton_server nlohmann_json::nlohmann_json)
target_compile_options(leapp_backend_common PRIVATE -Wall -Wextra -Wpedantic)
