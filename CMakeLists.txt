cmake_minimum_required(VERSION 3.20)
project(evrptw_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)
set(
  EVRPTW_SANITIZER
  "none"
  CACHE STRING
  "Native sanitizer profile: none, address-undefined, or thread"
)
set_property(
  CACHE EVRPTW_SANITIZER
  PROPERTY STRINGS none address-undefined thread
)

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

if(NOT EVRPTW_SANITIZER STREQUAL "none")
  if(MSVC)
    message(FATAL_ERROR "EVRPTW_SANITIZER profiles require GCC or Clang")
  elseif(EVRPTW_SANITIZER STREQUAL "address-undefined")
    target_compile_options(
      _core PRIVATE -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined
    )
    target_link_options(_core PRIVATE -fsanitize=address,undefined)
  elseif(EVRPTW_SANITIZER STREQUAL "thread")
    target_compile_options(
      _core PRIVATE -O1 -g -fno-omit-frame-pointer -fsanitize=thread
    )
    target_link_options(_core PRIVATE -fsanitize=thread)
  else()
    message(
      FATAL_ERROR
      "EVRPTW_SANITIZER must be none, address-undefined, or thread"
    )
  endif()
endif()

install(TARGETS _core DESTINATION evrptw)
