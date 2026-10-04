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
