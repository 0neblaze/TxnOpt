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
option(
  EVRPTW_ALLOW_DIRTY_BUILD
  "Allow a development-only native build from a dirty tracked source tree"
  OFF
)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(EVRPTW_ATTEST_OVERRIDE_ARGUMENT)
if(EVRPTW_ALLOW_DIRTY_BUILD)
  list(APPEND EVRPTW_ATTEST_OVERRIDE_ARGUMENT --development-override)
endif()
execute_process(
  COMMAND
    "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
    inspect
    --root "${CMAKE_CURRENT_SOURCE_DIR}"
    ${EVRPTW_ATTEST_OVERRIDE_ARGUMENT}
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE EVRPTW_BUILD_ATTESTATION_JSON
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE EVRPTW_BUILD_ATTESTATION_ERROR
  RESULT_VARIABLE EVRPTW_BUILD_ATTESTATION_RESULT
)
if(NOT EVRPTW_BUILD_ATTESTATION_RESULT EQUAL 0)
  message(FATAL_ERROR "${EVRPTW_BUILD_ATTESTATION_ERROR}")
endif()
string(JSON EVRPTW_BUILD_GIT_REVISION GET "${EVRPTW_BUILD_ATTESTATION_JSON}" revision)
string(JSON EVRPTW_BUILD_GIT_TREE GET "${EVRPTW_BUILD_ATTESTATION_JSON}" git_tree)
string(
  JSON EVRPTW_BUILD_SOURCE_MANIFEST_SHA256
  GET "${EVRPTW_BUILD_ATTESTATION_JSON}" source_manifest_sha256
)
string(
  JSON EVRPTW_BUILD_TRACKED_FILE_COUNT
  GET "${EVRPTW_BUILD_ATTESTATION_JSON}" tracked_file_count
)
string(JSON EVRPTW_BUILD_SOURCE_DIRTY_JSON GET "${EVRPTW_BUILD_ATTESTATION_JSON}" source_dirty)
string(
  JSON EVRPTW_BUILD_DEVELOPMENT_OVERRIDE_JSON
  GET "${EVRPTW_BUILD_ATTESTATION_JSON}" development_override
)
if(EVRPTW_BUILD_SOURCE_DIRTY_JSON)
  set(EVRPTW_BUILD_SOURCE_DIRTY 1)
  set(EVRPTW_BUILD_SOURCE_DIRTY_TEXT true)
else()
  set(EVRPTW_BUILD_SOURCE_DIRTY 0)
  set(EVRPTW_BUILD_SOURCE_DIRTY_TEXT false)
endif()
if(EVRPTW_BUILD_DEVELOPMENT_OVERRIDE_JSON)
  set(EVRPTW_BUILD_DEVELOPMENT_OVERRIDE 1)
  set(EVRPTW_BUILD_DEVELOPMENT_OVERRIDE_TEXT true)
else()
  set(EVRPTW_BUILD_DEVELOPMENT_OVERRIDE 0)
  set(EVRPTW_BUILD_DEVELOPMENT_OVERRIDE_TEXT false)
endif()

if(EVRPTW_ALLOW_DIRTY_BUILD)
  set(EVRPTW_NATIVE_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
  set(EVRPTW_BUILD_CPP_SOURCE_KIND "working_tree_override")
else()
  set(
    EVRPTW_NATIVE_SOURCE_SNAPSHOT
    "${CMAKE_CURRENT_BINARY_DIR}/attested-source-${EVRPTW_BUILD_SOURCE_MANIFEST_SHA256}"
  )
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
      materialize-cpp
      --root "${CMAKE_CURRENT_SOURCE_DIR}"
      --revision "${EVRPTW_BUILD_GIT_REVISION}"
      --destination "${EVRPTW_NATIVE_SOURCE_SNAPSHOT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    ERROR_VARIABLE EVRPTW_NATIVE_SNAPSHOT_ERROR
    RESULT_VARIABLE EVRPTW_NATIVE_SNAPSHOT_RESULT
  )
  if(NOT EVRPTW_NATIVE_SNAPSHOT_RESULT EQUAL 0)
    message(FATAL_ERROR "${EVRPTW_NATIVE_SNAPSHOT_ERROR}")
  endif()
  set(EVRPTW_NATIVE_CPP_DIR "${EVRPTW_NATIVE_SOURCE_SNAPSHOT}/cpp")
  set(EVRPTW_BUILD_CPP_SOURCE_KIND "git_blob_snapshot")
endif()

pybind11_add_module(_core "${EVRPTW_NATIVE_CPP_DIR}/evrptw_core.cpp")
target_include_directories(_core PRIVATE "${EVRPTW_NATIVE_CPP_DIR}")
if(UNIX AND NOT APPLE)
  add_executable(
    _native_host_scheduler
    "${EVRPTW_NATIVE_CPP_DIR}/native_host_scheduler.cpp"
  )
  target_include_directories(
    _native_host_scheduler PRIVATE "${EVRPTW_NATIVE_CPP_DIR}"
  )
endif()

add_custom_target(
  native_source_attestation_check ALL
  COMMAND
    "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
    verify
    --root "${CMAKE_CURRENT_SOURCE_DIR}"
    ${EVRPTW_ATTEST_OVERRIDE_ARGUMENT}
    --expected-revision "${EVRPTW_BUILD_GIT_REVISION}"
    --expected-tree "${EVRPTW_BUILD_GIT_TREE}"
    --expected-manifest-sha256 "${EVRPTW_BUILD_SOURCE_MANIFEST_SHA256}"
    --expected-source-dirty "${EVRPTW_BUILD_SOURCE_DIRTY_TEXT}"
    --expected-development-override "${EVRPTW_BUILD_DEVELOPMENT_OVERRIDE_TEXT}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Verifying native source-byte attestation"
  VERBATIM
)
if(NOT EVRPTW_ALLOW_DIRTY_BUILD)
  add_custom_command(
    TARGET native_source_attestation_check POST_BUILD
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
      verify-materialized-cpp
      --root "${CMAKE_CURRENT_SOURCE_DIR}"
      --revision "${EVRPTW_BUILD_GIT_REVISION}"
      --destination "${EVRPTW_NATIVE_SOURCE_SNAPSHOT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Verifying commit-materialized native C++ sources"
    VERBATIM
  )
endif()
add_dependencies(_core native_source_attestation_check)
if(TARGET _native_host_scheduler)
  add_dependencies(_native_host_scheduler native_source_attestation_check)
endif()

function(evrptw_add_post_build_attestation target_name)
  add_custom_command(
    TARGET ${target_name} POST_BUILD
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
      verify
      --root "${CMAKE_CURRENT_SOURCE_DIR}"
      ${EVRPTW_ATTEST_OVERRIDE_ARGUMENT}
      --expected-revision "${EVRPTW_BUILD_GIT_REVISION}"
      --expected-tree "${EVRPTW_BUILD_GIT_TREE}"
      --expected-manifest-sha256 "${EVRPTW_BUILD_SOURCE_MANIFEST_SHA256}"
      --expected-source-dirty "${EVRPTW_BUILD_SOURCE_DIRTY_TEXT}"
      --expected-development-override "${EVRPTW_BUILD_DEVELOPMENT_OVERRIDE_TEXT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Rechecking native source-byte attestation after linking"
    VERBATIM
  )
  if(NOT EVRPTW_ALLOW_DIRTY_BUILD)
    add_custom_command(
      TARGET ${target_name} POST_BUILD
      COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
        verify-materialized-cpp
        --root "${CMAKE_CURRENT_SOURCE_DIR}"
        --revision "${EVRPTW_BUILD_GIT_REVISION}"
        --destination "${EVRPTW_NATIVE_SOURCE_SNAPSHOT}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMENT "Rechecking commit-materialized C++ sources after linking"
      VERBATIM
    )
  endif()
endfunction()
evrptw_add_post_build_attestation(_core)
if(TARGET _native_host_scheduler)
  evrptw_add_post_build_attestation(_native_host_scheduler)
endif()

set(EVRPTW_BUILD_ATTESTATION_DEFINITIONS
  EVRPTW_BUILD_GIT_REVISION="${EVRPTW_BUILD_GIT_REVISION}"
  EVRPTW_BUILD_GIT_TREE="${EVRPTW_BUILD_GIT_TREE}"
  EVRPTW_BUILD_SOURCE_MANIFEST_SHA256="${EVRPTW_BUILD_SOURCE_MANIFEST_SHA256}"
  EVRPTW_BUILD_TRACKED_FILE_COUNT=${EVRPTW_BUILD_TRACKED_FILE_COUNT}
  EVRPTW_BUILD_SOURCE_DIRTY=${EVRPTW_BUILD_SOURCE_DIRTY}
  EVRPTW_BUILD_DEVELOPMENT_OVERRIDE=${EVRPTW_BUILD_DEVELOPMENT_OVERRIDE}
  EVRPTW_BUILD_CPP_SOURCE_KIND="${EVRPTW_BUILD_CPP_SOURCE_KIND}"
)
target_compile_definitions(_core PRIVATE ${EVRPTW_BUILD_ATTESTATION_DEFINITIONS})
if(TARGET _native_host_scheduler)
  target_compile_definitions(
    _native_host_scheduler PRIVATE ${EVRPTW_BUILD_ATTESTATION_DEFINITIONS}
  )
endif()

if(MSVC)
  target_compile_options(_core PRIVATE /O2)
else()
  target_compile_options(_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
  if(TARGET _native_host_scheduler)
    target_compile_options(
      _native_host_scheduler PRIVATE -O3 -Wall -Wextra -Wpedantic
    )
  endif()
endif()

if(NOT EVRPTW_SANITIZER STREQUAL "none")
  if(MSVC)
    message(FATAL_ERROR "EVRPTW_SANITIZER profiles require GCC or Clang")
  elseif(EVRPTW_SANITIZER STREQUAL "address-undefined")
    target_compile_options(
      _core PRIVATE -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined
    )
    target_link_options(_core PRIVATE -fsanitize=address,undefined)
    if(TARGET _native_host_scheduler)
      target_compile_options(
        _native_host_scheduler
        PRIVATE -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined
      )
      target_link_options(
        _native_host_scheduler PRIVATE -fsanitize=address,undefined
      )
    endif()
  elseif(EVRPTW_SANITIZER STREQUAL "thread")
    target_compile_options(
      _core PRIVATE -O1 -g -fno-omit-frame-pointer -fsanitize=thread
    )
    target_link_options(_core PRIVATE -fsanitize=thread)
    if(TARGET _native_host_scheduler)
      target_compile_options(
        _native_host_scheduler
        PRIVATE -O1 -g -fno-omit-frame-pointer -fsanitize=thread
      )
      target_link_options(_native_host_scheduler PRIVATE -fsanitize=thread)
    endif()
    add_executable(
      stage052_native_concurrency_tsan
      "${EVRPTW_NATIVE_CPP_DIR}/native_concurrency_tsan.cpp"
    )
    target_include_directories(
      stage052_native_concurrency_tsan PRIVATE "${EVRPTW_NATIVE_CPP_DIR}"
    )
    add_dependencies(
      stage052_native_concurrency_tsan native_source_attestation_check
    )
    evrptw_add_post_build_attestation(stage052_native_concurrency_tsan)
    target_compile_options(
      stage052_native_concurrency_tsan
      PRIVATE -O1 -g -fno-omit-frame-pointer -fsanitize=thread
    )
    target_link_options(stage052_native_concurrency_tsan PRIVATE -fsanitize=thread)
  else()
    message(
      FATAL_ERROR
      "EVRPTW_SANITIZER must be none, address-undefined, or thread"
    )
  endif()
endif()

install(TARGETS _core DESTINATION evrptw)
if(TARGET _native_host_scheduler)
  install(TARGETS _native_host_scheduler DESTINATION evrptw)
endif()
