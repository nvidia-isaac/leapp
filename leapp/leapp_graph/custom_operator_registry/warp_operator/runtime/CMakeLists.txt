cmake_minimum_required(VERSION 3.18)
project(leapp_warp_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(CONFIG IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG} "${CMAKE_BINARY_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG} "${CMAKE_BINARY_DIR}")
endforeach()

option(LEAPP_WARP_BUILD_ONNX "Build ONNX Runtime custom op" ON)
option(LEAPP_WARP_BUILD_TORCH "Build PyTorch dispatcher extension" ON)

include(GNUInstallDirs)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT DEFINED CMAKE_INSTALL_RPATH)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED ENV{CUDAToolkit_ROOT})
    set(CUDAToolkit_ROOT "$ENV{CUDAToolkit_ROOT}")
endif()
if(CUDAToolkit_ROOT AND NOT EXISTS "${CUDAToolkit_ROOT}/bin/nvcc${CMAKE_EXECUTABLE_SUFFIX}")
    message(FATAL_ERROR "CUDA compiler not found under CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
message(STATUS "Warp CUDA toolkit: ${CUDAToolkit_LIBRARY_ROOT} (${CUDAToolkit_VERSION})")

function(leapp_warp_python OUTPUT CODE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "${CODE}"
        OUTPUT_VARIABLE VALUE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    set(${OUTPUT} "${VALUE}" PARENT_SCOPE)
endfunction()

set(WARP_ROOT "$ENV{WARP_ROOT}" CACHE PATH "Warp SDK with native/ and bin/")
if(NOT WARP_ROOT)
    leapp_warp_python(WARP_ROOT "import os, warp; print(os.path.dirname(warp.__file__))")
endif()
set(WARP_NATIVE_DIR "${WARP_ROOT}/native")
set(WARP_BIN_DIR "${WARP_ROOT}/bin")
if(WIN32)
    set(WARP_SHARED_LIBRARY "${WARP_BIN_DIR}/warp.dll")
    set(WARP_IMPORT_LIBRARY "$ENV{LEAPP_WARP_IMPORT_LIBRARY}" CACHE FILEPATH "Warp import library")
    if(NOT WARP_IMPORT_LIBRARY)
        set(WARP_IMPORT_LIBRARY "${WARP_BIN_DIR}/warp.lib")
    endif()
elseif(APPLE)
    set(WARP_SHARED_LIBRARY "${WARP_BIN_DIR}/libwarp.dylib")
else()
    set(WARP_SHARED_LIBRARY "${WARP_BIN_DIR}/warp.so")
endif()

if(NOT EXISTS "${WARP_SHARED_LIBRARY}")
    message(FATAL_ERROR "Warp shared library not found: ${WARP_SHARED_LIBRARY}")
endif()
if(NOT EXISTS "${WARP_NATIVE_DIR}/apic.h")
    message(FATAL_ERROR "Warp APIC header not found: ${WARP_NATIVE_DIR}/apic.h. Install a Warp build with APIC support (Warp 1.13+ expected).")
endif()

message(STATUS "Warp native library: ${WARP_SHARED_LIBRARY}")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Warp's wheel ships warp.so without a SONAME or the usual lib prefix.
    # Exact-filename linking avoids recording the absolute SDK path in DT_NEEDED.
    add_library(warp_native INTERFACE)
    target_link_directories(warp_native INTERFACE "${WARP_BIN_DIR}")
    target_link_libraries(warp_native INTERFACE "-l:warp.so")
else()
    add_library(warp_native SHARED IMPORTED)
    set_target_properties(warp_native PROPERTIES IMPORTED_LOCATION "${WARP_SHARED_LIBRARY}")
    if(WIN32)
        if(NOT EXISTS "${WARP_IMPORT_LIBRARY}")
            message(FATAL_ERROR "Warp import library not found: ${WARP_IMPORT_LIBRARY}. Set WARP_IMPORT_LIBRARY to a valid warp.lib.")
        endif()
        set_target_properties(warp_native PROPERTIES IMPORTED_IMPLIB "${WARP_IMPORT_LIBRARY}")
    endif()
endif()

set(LEAPP_WARP_CORE_SOURCES
    core/runtime_metadata.cc
    core/temp_bundle.cc
    core/tensor_view.cc
    core/warp_apic_runner.cc
    core/wrpb_archive.cc
)

add_library(leapp_warp_runtime_core STATIC ${LEAPP_WARP_CORE_SOURCES})
target_include_directories(leapp_warp_runtime_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/core"
    "${WARP_NATIVE_DIR}"
)
target_link_libraries(leapp_warp_runtime_core PUBLIC warp_native CUDA::cudart Threads::Threads)
# Use CUDA vector types when Warp headers are included with the CUDA SDK.
target_compile_definitions(leapp_warp_runtime_core PRIVATE WP_ENABLE_CUDA=1)
set_target_properties(leapp_warp_runtime_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LEAPP_WARP_BUILD_ONNX)
    set(ORT_INCLUDE_DIR "$ENV{ORT_INCLUDE_DIR}" CACHE PATH "ONNX Runtime C API headers")
    set(LEAPP_WARP_ONNXRUNTIME_VERSION "" CACHE STRING "Target ONNX Runtime header version (defaults to the Python package)")
    if(NOT ORT_INCLUDE_DIR)
        set(ORT_HEADER_VERSION "${LEAPP_WARP_ONNXRUNTIME_VERSION}")
        if(NOT ORT_HEADER_VERSION)
            leapp_warp_python(ORT_HEADER_VERSION "import onnxruntime; print(onnxruntime.__version__)")
        endif()
        set(ORT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/onnxruntime_headers/${ORT_HEADER_VERSION}")
        file(MAKE_DIRECTORY "${ORT_INCLUDE_DIR}")
        foreach(ORT_HEADER IN ITEMS
            onnxruntime_c_api.h
            onnxruntime_ep_c_api.h
            onnxruntime_error_code.h
        )
            set(ORT_HEADER_PATH "${ORT_INCLUDE_DIR}/${ORT_HEADER}")
            if(NOT EXISTS "${ORT_HEADER_PATH}")
                file(DOWNLOAD
                    "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ORT_HEADER_VERSION}/include/onnxruntime/core/session/${ORT_HEADER}"
                    "${ORT_HEADER_PATH}"
                    STATUS ORT_HEADER_DOWNLOAD_STATUS
                    TLS_VERIFY ON
                )
                list(GET ORT_HEADER_DOWNLOAD_STATUS 0 ORT_HEADER_DOWNLOAD_CODE)
                list(GET ORT_HEADER_DOWNLOAD_STATUS 1 ORT_HEADER_DOWNLOAD_MESSAGE)
                if(NOT ORT_HEADER_DOWNLOAD_CODE EQUAL 0)
                    # Split out of onnxruntime_c_api.h in 1.29; older releases 404.
                    if(ORT_HEADER STREQUAL "onnxruntime_error_code.h")
                        file(REMOVE "${ORT_HEADER_PATH}")
                        message(STATUS
                            "Optional ${ORT_HEADER} is not available for "
                            "ONNX Runtime ${ORT_HEADER_VERSION}; skipping.")
                    else()
                        message(FATAL_ERROR "Failed to download ${ORT_HEADER}: ${ORT_HEADER_DOWNLOAD_MESSAGE}")
                    endif()
                endif()
            endif()
        endforeach()
    endif()
    foreach(ORT_HEADER IN ITEMS onnxruntime_c_api.h onnxruntime_ep_c_api.h)
        if(NOT EXISTS "${ORT_INCLUDE_DIR}/${ORT_HEADER}")
            message(FATAL_ERROR "ONNX Runtime header not found: ${ORT_INCLUDE_DIR}/${ORT_HEADER}. Set ORT_INCLUDE_DIR to the target SDK headers.")
        endif()
    endforeach()
    message(STATUS "Warp ONNX adapter headers: ${ORT_INCLUDE_DIR}")

    add_library(leapp_wrp_onnx_custom_op SHARED onnx/ort_wrp_runner_op.cc)
    target_include_directories(leapp_wrp_onnx_custom_op PRIVATE "${ORT_INCLUDE_DIR}")
    target_link_libraries(leapp_wrp_onnx_custom_op PRIVATE leapp_warp_runtime_core)
    if(NOT WIN32)
        set_property(TARGET leapp_wrp_onnx_custom_op APPEND PROPERTY
            BUILD_RPATH "${WARP_BIN_DIR};${CUDAToolkit_LIBRARY_DIR}")
    endif()
endif()

if(LEAPP_WARP_BUILD_TORCH)
    if(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for Torch-enabled builds")
    endif()
    if(NOT Torch_DIR AND DEFINED ENV{Torch_DIR})
        set(Torch_DIR "$ENV{Torch_DIR}")
    endif()
    if(Torch_DIR AND NOT EXISTS "${Torch_DIR}/TorchConfig.cmake")
        message(FATAL_ERROR "TorchConfig.cmake not found under Torch_DIR: ${Torch_DIR}")
    endif()
    find_package(Torch QUIET CONFIG)
    if(NOT Torch_FOUND)
        leapp_warp_python(TORCH_CMAKE_PREFIX_PATH "import torch; print(torch.utils.cmake_prefix_path)")
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
        find_package(Torch REQUIRED CONFIG)
    endif()
    message(STATUS "Warp Torch SDK: ${Torch_DIR}")
    add_library(leapp_wrp_torch_custom_op SHARED torch/torch_warp_runner_op.cc)
    target_link_libraries(leapp_wrp_torch_custom_op PRIVATE leapp_warp_runtime_core "${TORCH_LIBRARIES}")
    target_include_directories(leapp_wrp_torch_custom_op PRIVATE "${TORCH_INCLUDE_DIRS}")
    if(NOT WIN32)
        set_property(TARGET leapp_wrp_torch_custom_op APPEND PROPERTY
            BUILD_RPATH "${WARP_BIN_DIR};${CUDAToolkit_LIBRARY_DIR}")
    endif()
endif()

foreach(ADAPTER IN ITEMS leapp_wrp_onnx_custom_op leapp_wrp_torch_custom_op)
    if(TARGET ${ADAPTER})
        install(TARGETS ${ADAPTER}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endforeach()
