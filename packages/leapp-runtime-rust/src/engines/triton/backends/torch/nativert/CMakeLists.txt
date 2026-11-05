# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build NativeRT separately when the PyTorch wheel omits it. This target uses
# the supplied source tree without changing it or installing dependencies.
cmake_minimum_required(VERSION 3.24)
project(leapp_nativert LANGUAGES CXX)

set(PYTORCH_SOURCE "" CACHE PATH "PyTorch v2.12.1 source with its submodules")
set(TORCH_ROOT "" CACHE PATH "Matching torch package or LibTorch SDK root")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The NativeRT source build requires Linux")
endif()
foreach(required IN ITEMS
    "${PYTORCH_SOURCE}/version.txt"
    "${PYTORCH_SOURCE}/torch/nativert/ModelRunner.cpp"
    "${PYTORCH_SOURCE}/third_party/nlohmann/include/nlohmann/json.hpp"
    "${PYTORCH_SOURCE}/third_party/concurrentqueue/moodycamel/concurrentqueue.h"
    "${TORCH_ROOT}/include/torch/headeronly/version.h"
    "${TORCH_ROOT}/lib/libtorch.so"
    "${TORCH_ROOT}/lib/libtorch_cpu.so"
    "${TORCH_ROOT}/lib/libc10.so")
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

file(GLOB_RECURSE nativert_sources CONFIGURE_DEPENDS
  "${PYTORCH_SOURCE}/torch/nativert/*.cpp")
# Raw exported programs use dispatcher kernels, without Python or AOT delegates.
list(FILTER nativert_sources EXCLUDE REGEX
  "/python/|CudaTriton|AOTInductorModelContainerCudaShim")
# Optional optimized kernels reference private ATen symbols absent from wheels.
# The empty static registry falls back to NativeRT's ordinary C10 dispatcher.
list(FILTER nativert_sources EXCLUDE REGEX
  "/(KernelRegistry|NativeKernels|GeneratedStaticDispatchKernels|GeneratedNativeStaticDispatchKernels)[.]cpp$")
add_library(leapp_nativert SHARED ${nativert_sources} dispatcher_registry.cc
  "${PYTORCH_SOURCE}/torch/csrc/inductor/aoti_torch/oss_proxy_executor.cpp")
target_compile_definitions(leapp_nativert PRIVATE
  _GLIBCXX_USE_CXX11_ABI=1 FMT_HEADER_ONLY)
target_include_directories(leapp_nativert PRIVATE
  "${TORCH_ROOT}/include"
  "${TORCH_ROOT}/include/torch/csrc/api/include"
  "${PYTORCH_SOURCE}"
  "${PYTORCH_SOURCE}/third_party/nlohmann/include"
  "${PYTORCH_SOURCE}/third_party/concurrentqueue")
target_link_libraries(leapp_nativert PRIVATE
  "${TORCH_ROOT}/lib/libtorch.so"
  "${TORCH_ROOT}/lib/libtorch_cpu.so"
  "${TORCH_ROOT}/lib/libc10.so" ${CMAKE_DL_LIBS})
target_link_options(leapp_nativert PRIVATE "LINKER:--no-undefined")
