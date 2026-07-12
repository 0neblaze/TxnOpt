cmake_minimum_required(VERSION 3.20)
project(evrptw_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core cpp/evrptw_core.cpp)

if(MSVC)
  target_compile_options(_core PRIVATE /O2)
else()
  target_compile_options(_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _core DESTINATION evrptw)
