cmake_minimum_required(VERSION 3.20)
project(evrptw_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core cpp/evrptw_core.cpp)

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE EVRPTW_BUILD_GIT_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE EVRPTW_GIT_REVISION_RESULT
)
if(NOT EVRPTW_GIT_REVISION_RESULT EQUAL 0)
  set(EVRPTW_BUILD_GIT_REVISION "unknown")
endif()
target_compile_definitions(
  _core PRIVATE EVRPTW_BUILD_GIT_REVISION="${EVRPTW_BUILD_GIT_REVISION}"
)

if(MSVC)
  target_compile_options(_core PRIVATE /O2)
else()
  target_compile_options(_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _core DESTINATION evrptw)
