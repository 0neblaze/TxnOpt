cmake_minimum_required(VERSION 3.20)
project(evrptw_core LANGUAGES CXX)

if(APPLE)
  enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core cpp/evrptw_core.cpp)

if(APPLE)
  target_sources(_core PRIVATE cpp/metal_batch.mm)
  target_link_libraries(_core PRIVATE "-framework Metal" "-framework Foundation")
  set_source_files_properties(cpp/metal_batch.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

if(MSVC)
  target_compile_options(_core PRIVATE /O2)
else()
  target_compile_options(_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _core DESTINATION evrptw)
