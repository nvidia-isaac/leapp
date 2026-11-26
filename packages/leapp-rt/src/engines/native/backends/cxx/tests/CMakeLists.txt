# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(leapp_native_backend_tests LANGUAGES CXX)

# Reuse the same tests for both source builds and installed SDK validation.
if(NOT TARGET lrt::native_backend)
  find_package(leapp_native_backend CONFIG REQUIRED)
endif()

add_library(leapp_plugin_fixture SHARED fixture.cc)
target_link_libraries(leapp_plugin_fixture PRIVATE lrt::native_backend)
target_compile_options(leapp_plugin_fixture PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(leapp_plugin_fixture PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins")

add_executable(leapp_plugin_types types_test.cc)
target_link_libraries(leapp_plugin_types PRIVATE lrt::native_backend)
target_compile_options(leapp_plugin_types PRIVATE -Wall -Wextra -Wpedantic -Werror)
enable_testing()
add_test(NAME native_plugin_types COMMAND leapp_plugin_types)
