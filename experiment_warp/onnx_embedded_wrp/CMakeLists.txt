cmake_minimum_required(VERSION 3.18)
project(onnx_embedded_wrp_op LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import os, onnxruntime; print(os.path.dirname(onnxruntime.__file__))"
    OUTPUT_VARIABLE ONNXRUNTIME_PACKAGE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import os, warp; print(os.path.dirname(warp.__file__))"
    OUTPUT_VARIABLE WARP_PACKAGE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import onnxruntime; print(onnxruntime.__version__)"
    OUTPUT_VARIABLE ONNXRUNTIME_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

set(ORT_CAPI_DIR "${ONNXRUNTIME_PACKAGE_DIR}/capi")
set(ORT_SHARED_LIBRARY "${ORT_CAPI_DIR}/libonnxruntime.so.${ONNXRUNTIME_VERSION}")
set(WARP_NATIVE_DIR "${WARP_PACKAGE_DIR}/native")
set(WARP_BIN_DIR "${WARP_PACKAGE_DIR}/bin")
set(WARP_SHARED_LIBRARY "${WARP_BIN_DIR}/warp.so")

if(NOT EXISTS "${ORT_SHARED_LIBRARY}")
    message(FATAL_ERROR "ONNX Runtime shared library not found: ${ORT_SHARED_LIBRARY}")
endif()

if(NOT EXISTS "${WARP_SHARED_LIBRARY}")
    message(FATAL_ERROR "Warp shared library not found: ${WARP_SHARED_LIBRARY}")
endif()

set(ORT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/onnxruntime_headers")
file(MAKE_DIRECTORY "${ORT_INCLUDE_DIR}")
foreach(ORT_HEADER IN ITEMS onnxruntime_c_api.h onnxruntime_ep_c_api.h)
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
            message(FATAL_ERROR "Failed to download ${ORT_HEADER}: ${ORT_HEADER_DOWNLOAD_MESSAGE}")
        endif()
    endif()
endforeach()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_SHARED_LIBRARY}"
)

add_library(warp_native SHARED IMPORTED)
set_target_properties(warp_native PROPERTIES
    IMPORTED_LOCATION "${WARP_SHARED_LIBRARY}"
)

add_library(wrp_onnx_custom_op SHARED wrp_runner_op.cc)
target_include_directories(wrp_onnx_custom_op PRIVATE
    "${ORT_INCLUDE_DIR}"
    "${WARP_NATIVE_DIR}"
)
target_link_libraries(wrp_onnx_custom_op PRIVATE onnxruntime warp_native CUDA::cudart dl pthread)
set_target_properties(wrp_onnx_custom_op PROPERTIES
    PREFIX "lib"
    BUILD_RPATH "${ORT_CAPI_DIR};${WARP_BIN_DIR};${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${ORT_CAPI_DIR};${WARP_BIN_DIR};${CUDAToolkit_LIBRARY_DIR}"
)
