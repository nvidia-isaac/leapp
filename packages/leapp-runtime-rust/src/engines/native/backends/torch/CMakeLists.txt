# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The native Torch backends currently require Linux")
endif()
set(TORCH_ROOT "$ENV{TORCH_ROOT}" CACHE PATH "CUDA-enabled LibTorch SDK or torch package")
set(TORCH_CXX11_ABI "1" CACHE STRING "LibTorch C++11 ABI (0 or 1)")
if(NOT TORCH_CXX11_ABI MATCHES "^[01]$")
  message(FATAL_ERROR "Set TORCH_CXX11_ABI to match the installed SDK (0 or 1)")
endif()
if(NOT EXISTS "${TORCH_ROOT}/include/torch/script.h")
  message(FATAL_ERROR "Supply TORCH_ROOT with include/ and lib/ from LibTorch")
endif()
find_package(CUDAToolkit REQUIRED)
add_library(leapp_native_torch INTERFACE)
target_compile_features(leapp_native_torch INTERFACE cxx_std_20)
target_compile_definitions(leapp_native_torch INTERFACE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_include_directories(leapp_native_torch SYSTEM INTERFACE
  "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include")
foreach(name IN ITEMS torch torch_cpu torch_cuda c10 c10_cuda)
  find_library(LEAPP_${name}_LIBRARY NAMES ${name}
    PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  target_link_libraries(leapp_native_torch INTERFACE "${LEAPP_${name}_LIBRARY}")
endforeach()
target_link_libraries(leapp_native_torch INTERFACE CUDA::cudart)
target_sources(leapp_native_torch INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/engine.cc")
target_compile_options(leapp_native_torch INTERFACE -Wall -Wextra -Wno-unused-parameter)
target_link_options(leapp_native_torch INTERFACE "LINKER:--no-undefined")
