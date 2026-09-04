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

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import os, warp; print(os.path.dirname(warp.__file__))"
    OUTPUT_VARIABLE WARP_PACKAGE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

set(WARP_NATIVE_DIR "${WARP_PACKAGE_DIR}/native")
set(WARP_BIN_DIR "${WARP_PACKAGE_DIR}/bin")
if(WIN32)
    set(WARP_SHARED_LIBRARY "${WARP_BIN_DIR}/warp.dll")
    set(WARP_IMPORT_LIBRARY "${WARP_BIN_DIR}/warp.lib" CACHE FILEPATH "Warp import library")
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

add_library(warp_native SHARED IMPORTED)
set_target_properties(warp_native PROPERTIES
    IMPORTED_LOCATION "${WARP_SHARED_LIBRARY}"
)
if(WIN32)
    if(NOT EXISTS "${WARP_IMPORT_LIBRARY}")
        message(FATAL_ERROR "Warp import library not found: ${WARP_IMPORT_LIBRARY}. Set WARP_IMPORT_LIBRARY to a valid warp.lib.")
    endif()
    set_target_properties(warp_native PROPERTIES
        IMPORTED_IMPLIB "${WARP_IMPORT_LIBRARY}"
    )
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
set_target_properties(leapp_warp_runtime_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LEAPP_WARP_BUILD_ONNX)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import os, onnxruntime; print(os.path.dirname(onnxruntime.__file__))"
        OUTPUT_VARIABLE ONNXRUNTIME_PACKAGE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import onnxruntime; print(onnxruntime.__version__)"
        OUTPUT_VARIABLE ONNXRUNTIME_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )

    set(ORT_CAPI_DIR "${ONNXRUNTIME_PACKAGE_DIR}/capi")
    if(WIN32)
        set(ORT_SHARED_LIBRARY "${ORT_CAPI_DIR}/onnxruntime.dll")
    elseif(APPLE)
        set(ORT_SHARED_LIBRARY "${ORT_CAPI_DIR}/libonnxruntime.${ONNXRUNTIME_VERSION}.dylib")
    else()
        set(ORT_SHARED_LIBRARY "${ORT_CAPI_DIR}/libonnxruntime.so.${ONNXRUNTIME_VERSION}")
    endif()
    if(NOT EXISTS "${ORT_SHARED_LIBRARY}")
        message(FATAL_ERROR "ONNX Runtime shared library not found: ${ORT_SHARED_LIBRARY}")
    endif()

    set(ORT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/onnxruntime_headers")
    file(MAKE_DIRECTORY "${ORT_INCLUDE_DIR}")
    foreach(ORT_HEADER IN ITEMS
        onnxruntime_c_api.h
        onnxruntime_ep_c_api.h
        onnxruntime_error_code.h
    )
        set(ORT_HEADER_PATH "${ORT_INCLUDE_DIR}/${ORT_HEADER}")
        if(NOT EXISTS "${ORT_HEADER_PATH}")
            file(DOWNLOAD
                "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNXRUNTIME_VERSION}/include/onnxruntime/core/session/${ORT_HEADER}"
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
                        "ONNX Runtime ${ONNXRUNTIME_VERSION}; skipping.")
                else()
                    message(FATAL_ERROR "Failed to download ${ORT_HEADER}: ${ORT_HEADER_DOWNLOAD_MESSAGE}")
                endif()
            endif()
        endif()
    endforeach()

    add_library(leapp_wrp_onnx_custom_op SHARED onnx/ort_wrp_runner_op.cc)
    target_include_directories(leapp_wrp_onnx_custom_op PRIVATE "${ORT_INCLUDE_DIR}")
    target_link_libraries(leapp_wrp_onnx_custom_op PRIVATE leapp_warp_runtime_core)
    if(NOT WIN32)
        set_target_properties(leapp_wrp_onnx_custom_op PROPERTIES
            BUILD_RPATH "${WARP_BIN_DIR};${CUDAToolkit_LIBRARY_DIR}"
            INSTALL_RPATH "${WARP_BIN_DIR};${CUDAToolkit_LIBRARY_DIR}"
        )
    endif()
endif()

if(LEAPP_WARP_BUILD_TORCH)
    if(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for Torch-enabled builds")
    endif()
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
    find_package(Torch REQUIRED)
    add_library(leapp_wrp_torch_custom_op SHARED torch/torch_warp_runner_op.cc)
    target_link_libraries(leapp_wrp_torch_custom_op PRIVATE leapp_warp_runtime_core "${TORCH_LIBRARIES}")
    target_include_directories(leapp_wrp_torch_custom_op PRIVATE "${TORCH_INCLUDE_DIRS}")
    if(NOT WIN32)
        set_target_properties(leapp_wrp_torch_custom_op PROPERTIES
            BUILD_RPATH "${WARP_BIN_DIR};${CUDAToolkit_LIBRARY_DIR}"
            INSTALL_RPATH "${WARP_BIN_DIR};${CUDAToolkit_LIBRARY_DIR}"
        )
    endif()
endif()
