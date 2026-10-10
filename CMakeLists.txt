cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(_connected src/conncomp/_connected.cpp)
target_compile_options(_connected PRIVATE -O3 -Wall)
target_link_libraries(_connected PRIVATE Threads::Threads)

install(TARGETS _connected DESTINATION ${SKBUILD_PROJECT_NAME})

# Optional CUDA module (block-based union-find labelling). Built only if a CUDA compiler is available; prefer the
# toolkit in /usr/local/cuda (the distribution's /usr/bin/nvcc may be too old for recent GPUs).
include(CheckLanguage)
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)  # must be set before enable_language (which sets a default)
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89;120")  # Turing, Ampere, Ada, Blackwell (sm_120)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(ccl_kernels STATIC src/conncomp/_cuda_ccl.cu)
    set_target_properties(ccl_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
    target_compile_options(ccl_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
    pybind11_add_module(_connected_cuda src/conncomp/_connected_cuda.cpp)
    target_link_libraries(_connected_cuda PRIVATE ccl_kernels CUDA::cudart_static)
    install(TARGETS _connected_cuda DESTINATION ${SKBUILD_PROJECT_NAME})
else()
    message(STATUS "No CUDA compiler found: building without the CUDA labelling module")
endif()
