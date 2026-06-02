cmake_minimum_required(VERSION 3.18)
project(warp_onnx_vs_cpp_benchmark LANGUAGES CXX)

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
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pathlib, site; paths=[]\nfor base in site.getsitepackages():\n p=pathlib.Path(base)/'nvidia'\n if p.exists():\n  paths += [str(x/'lib') for x in p.iterdir() if (x/'lib').exists()]\nprint(';'.join(paths))"
    OUTPUT_VARIABLE NVIDIA_PY_LIB_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

set(ORT_CAPI_DIR "${ONNXRUNTIME_PACKAGE_DIR}/capi")
set(ORT_SHARED_LIBRARY "${ORT_CAPI_DIR}/libonnxruntime.so.${ONNXRUNTIME_VERSION}")
set(WARP_NATIVE_DIR "${WARP_PACKAGE_DIR}/native")
set(WARP_BIN_DIR "${WARP_PACKAGE_DIR}/bin")
set(WARP_SHARED_LIBRARY "${WARP_BIN_DIR}/warp.so")
set(ORT_SONAME_LINK "${CMAKE_BINARY_DIR}/libonnxruntime.so.1")

if(NOT EXISTS "${ORT_SHARED_LIBRARY}")
    message(FATAL_ERROR "ONNX Runtime shared library not found: ${ORT_SHARED_LIBRARY}")
endif()
if(NOT EXISTS "${WARP_SHARED_LIBRARY}")
    message(FATAL_ERROR "Warp shared library not found: ${WARP_SHARED_LIBRARY}")
endif()

set(ORT_INCLUDE_DIR "${CMAKE_BINARY_DIR}/onnxruntime_headers")
file(MAKE_DIRECTORY "${ORT_INCLUDE_DIR}")
foreach(ORT_HEADER IN ITEMS
        onnxruntime_c_api.h
        onnxruntime_ep_c_api.h
        onnxruntime_cxx_api.h
        onnxruntime_cxx_inline.h
        onnxruntime_float16.h)
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
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ORT_SHARED_LIBRARY}")
file(CREATE_LINK "${ORT_SHARED_LIBRARY}" "${ORT_SONAME_LINK}" SYMBOLIC)
file(GLOB ORT_PROVIDER_LIBS "${ORT_CAPI_DIR}/libonnxruntime_providers*.so")
foreach(ORT_PROVIDER_LIB IN LISTS ORT_PROVIDER_LIBS)
    get_filename_component(ORT_PROVIDER_NAME "${ORT_PROVIDER_LIB}" NAME)
    file(CREATE_LINK "${ORT_PROVIDER_LIB}" "${CMAKE_BINARY_DIR}/${ORT_PROVIDER_NAME}" SYMBOLIC)
endforeach()
foreach(NVIDIA_PY_LIB_DIR IN LISTS NVIDIA_PY_LIB_DIRS)
    file(GLOB NVIDIA_PY_LIBS "${NVIDIA_PY_LIB_DIR}/*.so*")
    foreach(NVIDIA_PY_LIB IN LISTS NVIDIA_PY_LIBS)
        get_filename_component(NVIDIA_PY_LIB_NAME "${NVIDIA_PY_LIB}" NAME)
        if(NOT EXISTS "${CMAKE_BINARY_DIR}/${NVIDIA_PY_LIB_NAME}")
            file(CREATE_LINK "${NVIDIA_PY_LIB}" "${CMAKE_BINARY_DIR}/${NVIDIA_PY_LIB_NAME}" SYMBOLIC)
        endif()
    endforeach()
endforeach()

add_library(warp_native SHARED IMPORTED)
set_target_properties(warp_native PROPERTIES IMPORTED_LOCATION "${WARP_SHARED_LIBRARY}")

set(COMMON_RPATH "${CMAKE_BINARY_DIR};${ORT_CAPI_DIR};${WARP_BIN_DIR};${CUDAToolkit_LIBRARY_DIR};${NVIDIA_PY_LIB_DIRS}")

add_library(benchmark_wrp_onnx_custom_op SHARED
    ../onnx_embedded_wrp/wrp_runner_op.cc
)
target_include_directories(benchmark_wrp_onnx_custom_op PRIVATE
    "${ORT_INCLUDE_DIR}"
    "${WARP_NATIVE_DIR}"
)
target_link_libraries(benchmark_wrp_onnx_custom_op PRIVATE onnxruntime warp_native CUDA::cudart dl pthread)
set_target_properties(benchmark_wrp_onnx_custom_op PROPERTIES
    PREFIX "lib"
    BUILD_RPATH "${COMMON_RPATH}"
    INSTALL_RPATH "${COMMON_RPATH}"
)

# CUDA-resident variant: keeps all per-node tensors on the GPU (CUDA EP) and
# moves data device->device on ORT's compute stream instead of host round-trips.
add_library(benchmark_wrp_onnx_cuda_op SHARED
    wrp_runner_cuda_op.cc
)
target_include_directories(benchmark_wrp_onnx_cuda_op PRIVATE
    "${ORT_INCLUDE_DIR}"
    "${WARP_NATIVE_DIR}"
)
target_link_libraries(benchmark_wrp_onnx_cuda_op PRIVATE onnxruntime warp_native CUDA::cudart dl pthread)
set_target_properties(benchmark_wrp_onnx_cuda_op PROPERTIES
    PREFIX "lib"
    BUILD_RPATH "${COMMON_RPATH}"
    INSTALL_RPATH "${COMMON_RPATH}"
)

add_executable(run_cpp_sequence run_cpp_sequence.cc)
target_include_directories(run_cpp_sequence PRIVATE
    "${ORT_INCLUDE_DIR}"
    "${WARP_NATIVE_DIR}"
)
target_link_libraries(run_cpp_sequence PRIVATE onnxruntime warp_native CUDA::cudart dl pthread)
set_target_properties(run_cpp_sequence PROPERTIES
    BUILD_RPATH "${COMMON_RPATH}"
    INSTALL_RPATH "${COMMON_RPATH}"
)

add_executable(run_embedded_onnx run_embedded_onnx.cc)
target_include_directories(run_embedded_onnx PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(run_embedded_onnx PRIVATE onnxruntime CUDA::cudart dl pthread)
set_target_properties(run_embedded_onnx PROPERTIES
    BUILD_RPATH "${COMMON_RPATH}"
    INSTALL_RPATH "${COMMON_RPATH}"
)
