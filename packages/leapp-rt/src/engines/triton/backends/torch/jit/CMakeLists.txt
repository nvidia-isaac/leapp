# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(leapp_torchscript_backend LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The TorchScript backend requires Linux")
endif()
add_subdirectory(../common common)
add_library(triton_leapp_torchscript SHARED
  src/backend.cc src/input_analysis.cc src/torchscript_instance.cc)
target_include_directories(triton_leapp_torchscript PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(triton_leapp_torchscript PRIVATE leapp_backend_common)
target_compile_options(triton_leapp_torchscript PRIVATE -Wall -Wextra -Wpedantic)
target_link_options(triton_leapp_torchscript PRIVATE "LINKER:--no-undefined")
set_target_properties(triton_leapp_torchscript PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/backends/leapp_torchscript")
install(TARGETS triton_leapp_torchscript LIBRARY DESTINATION backends/leapp_torchscript)
include(CTest)
if(BUILD_TESTING)
  add_executable(torchscript_backend_test
    tests/backend_test.cc src/input_analysis.cc)
  target_include_directories(torchscript_backend_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(torchscript_backend_test PRIVATE leapp_backend_common)
  target_compile_options(torchscript_backend_test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME torchscript_backend COMMAND torchscript_backend_test)
endif()
