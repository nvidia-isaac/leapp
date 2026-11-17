# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(leapp_native_backend VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
option(LEAPP_INSTALL_NATIVE_BACKEND_SDK "Install the standalone backend SDK" ${PROJECT_IS_TOP_LEVEL})

if(NOT TARGET lrt::native_backend)
  add_library(leapp_native_backend INTERFACE)
  add_library(lrt::native_backend ALIAS leapp_native_backend)
  set_target_properties(leapp_native_backend PROPERTIES EXPORT_NAME native_backend)
  target_compile_features(leapp_native_backend INTERFACE cxx_std_20)
  target_include_directories(leapp_native_backend INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../dlpack/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  # Compile with each plugin's STL ABI settings, including LibTorch's ABI switch.
  target_sources(leapp_native_backend INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/adapter.cc>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_DATADIR}/leapp-native-backend/adapter.cc>")
  if(LEAPP_INSTALL_NATIVE_BACKEND_SDK)
    install(TARGETS leapp_native_backend EXPORT leapp_native_backendTargets)
    install(EXPORT leapp_native_backendTargets NAMESPACE lrt::
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/leapp_native_backend")
    install(DIRECTORY ../include/leapp
      ../../../../dlpack/include/dlpack DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(FILES src/adapter.cc DESTINATION "${CMAKE_INSTALL_DATADIR}/leapp-native-backend")
    configure_package_config_file(cmake/leapp_native_backendConfig.cmake.in
      "${CMAKE_CURRENT_BINARY_DIR}/leapp_native_backendConfig.cmake"
      INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/leapp_native_backend")
    write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/leapp_native_backendConfigVersion.cmake"
      VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/leapp_native_backendConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/leapp_native_backendConfigVersion.cmake"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/leapp_native_backend")
  endif()
endif()

option(LEAPP_NATIVE_BACKEND_TESTS "Build backend SDK contract tests" ${PROJECT_IS_TOP_LEVEL})
if(LEAPP_NATIVE_BACKEND_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
