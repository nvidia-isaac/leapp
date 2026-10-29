cmake_minimum_required(VERSION 3.22)
project(leapp_runtime VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

find_package(Threads REQUIRED)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(RUST_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(RUST_LIBRARY "${RUST_TARGET_DIR}/release/libleapp_runtime_cxx.a")

# Cargo tracks Rust sources and dependencies; this also refreshes generated headers.
add_custom_target(leapp_runtime_rust
  COMMAND "${CMAKE_COMMAND}" -E env --unset=MAKEFLAGS --unset=MFLAGS
          "${CARGO_EXECUTABLE}" build --locked --release
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
          --target-dir "${RUST_TARGET_DIR}"
  BYPRODUCTS "${RUST_LIBRARY}"
    "${RUST_TARGET_DIR}/cxxbridge/leapp-runtime-cxx/src/bridge.rs.h"
  VERBATIM)

add_library(leapp_runtime SHARED src/bundle.cc)
add_library(lrt::runtime ALIAS leapp_runtime)
add_dependencies(leapp_runtime leapp_runtime_rust)
set_target_properties(leapp_runtime PROPERTIES EXPORT_NAME runtime
  VERSION ${PROJECT_VERSION} SOVERSION 0)
target_compile_features(leapp_runtime PUBLIC cxx_std_20)
target_include_directories(leapp_runtime
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${RUST_TARGET_DIR}/cxxbridge")

target_link_libraries(leapp_runtime PRIVATE "${RUST_LIBRARY}"
  Threads::Threads ${CMAKE_DL_LIBS} m)
if(APPLE)
  target_link_libraries(leapp_runtime PRIVATE iconv)
elseif(UNIX)
  target_link_libraries(leapp_runtime PRIVATE rt util)
  target_link_options(leapp_runtime PRIVATE "-Wl,--exclude-libs,ALL" "-Wl,--no-undefined")
endif()

target_compile_options(leapp_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror -fno-exceptions)

install(TARGETS leapp_runtime EXPORT leapp_runtimeTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/leapp DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT leapp_runtimeTargets NAMESPACE lrt::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/leapp_runtime")

configure_package_config_file(cmake/leapp_runtimeConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/leapp_runtimeConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/leapp_runtime")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/leapp_runtimeConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/leapp_runtimeConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/leapp_runtimeConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/leapp_runtime")

if(BUILD_TESTING)
  add_executable(leapp_bundle_test tests/bundle_test.cc)
  target_link_libraries(leapp_bundle_test PRIVATE lrt::runtime Threads::Threads)
  target_compile_options(leapp_bundle_test PRIVATE
    -Wall -Wextra -Wpedantic -Werror -fno-exceptions)
  add_test(NAME leapp_bundle COMMAND leapp_bundle_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/pipeline.yaml")
endif()
