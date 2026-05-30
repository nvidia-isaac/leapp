cmake_minimum_required(VERSION 3.18)
project(warp_apic_cpp_replay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import os, warp; print(os.path.dirname(warp.__file__))"
    OUTPUT_VARIABLE WARP_PACKAGE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

set(WARP_NATIVE_DIR "${WARP_PACKAGE_DIR}/native")
set(WARP_BIN_DIR "${WARP_PACKAGE_DIR}/bin")
set(WARP_SHARED_LIBRARY "${WARP_BIN_DIR}/warp.so")

if(NOT EXISTS "${WARP_SHARED_LIBRARY}")
    message(FATAL_ERROR "Warp shared library not found: ${WARP_SHARED_LIBRARY}")
endif()

add_library(warp_native SHARED IMPORTED)
set_target_properties(warp_native PROPERTIES
    IMPORTED_LOCATION "${WARP_SHARED_LIBRARY}"
)

add_executable(cpp_replay cpp_replay.cpp)
target_include_directories(cpp_replay PRIVATE "${WARP_NATIVE_DIR}")
target_link_libraries(cpp_replay PRIVATE warp_native dl pthread)
set_target_properties(cpp_replay PROPERTIES
    BUILD_RPATH "${WARP_BIN_DIR}"
    INSTALL_RPATH "${WARP_BIN_DIR}"
)
