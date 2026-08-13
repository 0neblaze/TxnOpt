cmake_minimum_required(VERSION 3.20)
project(txnopt_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)
set(
  _TXNOPT_EFFECTIVE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}"
)
string(
  TOLOWER
  "${_TXNOPT_EFFECTIVE_CXX_FLAGS}"
  _TXNOPT_EFFECTIVE_CXX_FLAGS_LOWER
)
if(
  _TXNOPT_EFFECTIVE_CXX_FLAGS_LOWER MATCHES "fast-math"
  OR _TXNOPT_EFFECTIVE_CXX_FLAGS_LOWER MATCHES "(^|[ ;])-ofast([ ;]|$)"
  OR _TXNOPT_EFFECTIVE_CXX_FLAGS_LOWER MATCHES "/fp:fast"
)
  message(
    FATAL_ERROR
    "fast-math and equivalent unsafe floating-point modes are forbidden"
  )
endif()
set(
  TXNOPT_SANITIZER
  "none"
  CACHE STRING
  "Native sanitizer profile: none, address-undefined, or thread"
)
set_property(
  CACHE TXNOPT_SANITIZER
  PROPERTY STRINGS none address-undefined thread
)
set(
  TXNOPT_PERFORMANCE_PROFILE
  "portable-o3"
  CACHE STRING
  "Native performance profile: portable-o3, portable-lto, or host-native-lto"
)
set_property(
  CACHE TXNOPT_PERFORMANCE_PROFILE
  PROPERTY STRINGS portable-o3 portable-lto host-native-lto
)
set(
  _TXNOPT_PERFORMANCE_PROFILES
  portable-o3
  portable-lto
  host-native-lto
)
if(NOT TXNOPT_PERFORMANCE_PROFILE IN_LIST _TXNOPT_PERFORMANCE_PROFILES)
  message(
    FATAL_ERROR
    "TXNOPT_PERFORMANCE_PROFILE must be portable-o3, portable-lto, or host-native-lto"
  )
endif()
set(TXNOPT_BUILD_INTERPROCEDURAL_OPTIMIZATION 0)
set(TXNOPT_BUILD_HOST_NATIVE 0)
if(TXNOPT_PERFORMANCE_PROFILE MATCHES "-lto$")
  if(NOT TXNOPT_SANITIZER STREQUAL "none")
    message(
      FATAL_ERROR
      "LTO performance profiles cannot be combined with sanitizer builds"
    )
  endif()
  include(CheckIPOSupported)
  check_ipo_supported(
    RESULT TXNOPT_IPO_SUPPORTED
    OUTPUT TXNOPT_IPO_DIAGNOSTIC
    LANGUAGES CXX
  )
  if(NOT TXNOPT_IPO_SUPPORTED)
    message(FATAL_ERROR "LTO is unavailable: ${TXNOPT_IPO_DIAGNOSTIC}")
  endif()
  set(TXNOPT_BUILD_INTERPROCEDURAL_OPTIMIZATION 1)
endif()
if(TXNOPT_PERFORMANCE_PROFILE STREQUAL "host-native-lto")
  if(
    NOT UNIX
    OR APPLE
    OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"
  )
    message(
      FATAL_ERROR
      "host-native-lto is restricted to Linux/WSL x86-64"
    )
  endif()
  set(TXNOPT_BUILD_HOST_NATIVE 1)
endif()
option(
  TXNOPT_ALLOW_DIRTY_BUILD
  "Allow a development-only native build from a dirty tracked source tree"
  OFF
)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(TXNOPT_ATTEST_OVERRIDE_ARGUMENT)
if(TXNOPT_ALLOW_DIRTY_BUILD)
  list(APPEND TXNOPT_ATTEST_OVERRIDE_ARGUMENT --development-override)
endif()
execute_process(
  COMMAND
    "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
    inspect
    --root "${CMAKE_CURRENT_SOURCE_DIR}"
    ${TXNOPT_ATTEST_OVERRIDE_ARGUMENT}
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE TXNOPT_BUILD_ATTESTATION_JSON
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE TXNOPT_BUILD_ATTESTATION_ERROR
  RESULT_VARIABLE TXNOPT_BUILD_ATTESTATION_RESULT
)
if(NOT TXNOPT_BUILD_ATTESTATION_RESULT EQUAL 0)
  message(FATAL_ERROR "${TXNOPT_BUILD_ATTESTATION_ERROR}")
endif()
string(JSON TXNOPT_BUILD_GIT_REVISION GET "${TXNOPT_BUILD_ATTESTATION_JSON}" revision)
string(JSON TXNOPT_BUILD_GIT_TREE GET "${TXNOPT_BUILD_ATTESTATION_JSON}" git_tree)
string(
  JSON TXNOPT_BUILD_SOURCE_MANIFEST_SHA256
  GET "${TXNOPT_BUILD_ATTESTATION_JSON}" source_manifest_sha256
)
string(
  JSON TXNOPT_BUILD_TRACKED_FILE_COUNT
  GET "${TXNOPT_BUILD_ATTESTATION_JSON}" tracked_file_count
)
string(JSON TXNOPT_BUILD_SOURCE_DIRTY_JSON GET "${TXNOPT_BUILD_ATTESTATION_JSON}" source_dirty)
string(
  JSON TXNOPT_BUILD_DEVELOPMENT_OVERRIDE_JSON
  GET "${TXNOPT_BUILD_ATTESTATION_JSON}" development_override
)
if(TXNOPT_BUILD_SOURCE_DIRTY_JSON)
  set(TXNOPT_BUILD_SOURCE_DIRTY 1)
  set(TXNOPT_BUILD_SOURCE_DIRTY_TEXT true)
else()
  set(TXNOPT_BUILD_SOURCE_DIRTY 0)
  set(TXNOPT_BUILD_SOURCE_DIRTY_TEXT false)
endif()
if(TXNOPT_BUILD_DEVELOPMENT_OVERRIDE_JSON)
  set(TXNOPT_BUILD_DEVELOPMENT_OVERRIDE 1)
  set(TXNOPT_BUILD_DEVELOPMENT_OVERRIDE_TEXT true)
else()
  set(TXNOPT_BUILD_DEVELOPMENT_OVERRIDE 0)
  set(TXNOPT_BUILD_DEVELOPMENT_OVERRIDE_TEXT false)
endif()

if(TXNOPT_ALLOW_DIRTY_BUILD)
  set(TXNOPT_NATIVE_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
  set(TXNOPT_BUILD_CPP_SOURCE_KIND "working_tree_override")
else()
  set(
    TXNOPT_NATIVE_SOURCE_SNAPSHOT
    "${CMAKE_CURRENT_BINARY_DIR}/attested-source-${TXNOPT_BUILD_SOURCE_MANIFEST_SHA256}"
  )
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
      materialize-cpp
      --root "${CMAKE_CURRENT_SOURCE_DIR}"
      --revision "${TXNOPT_BUILD_GIT_REVISION}"
      --destination "${TXNOPT_NATIVE_SOURCE_SNAPSHOT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    ERROR_VARIABLE TXNOPT_NATIVE_SNAPSHOT_ERROR
    RESULT_VARIABLE TXNOPT_NATIVE_SNAPSHOT_RESULT
  )
  if(NOT TXNOPT_NATIVE_SNAPSHOT_RESULT EQUAL 0)
    message(FATAL_ERROR "${TXNOPT_NATIVE_SNAPSHOT_ERROR}")
  endif()
  set(TXNOPT_NATIVE_CPP_DIR "${TXNOPT_NATIVE_SOURCE_SNAPSHOT}/cpp")
  set(TXNOPT_BUILD_CPP_SOURCE_KIND "git_blob_snapshot")
endif()

# Disable pybind11's implicit release-mode LTO/strip extras. The attested
# TxnOpt profile below is the sole owner of those build choices.
pybind11_add_module(
  _native
  MODULE
  NO_EXTRAS
  "${TXNOPT_NATIVE_CPP_DIR}/txnopt_core/python_module.cpp"
)
target_include_directories(_native PRIVATE "${TXNOPT_NATIVE_CPP_DIR}")

add_custom_target(
  native_source_attestation_check ALL
  COMMAND
    "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
    verify
    --root "${CMAKE_CURRENT_SOURCE_DIR}"
    ${TXNOPT_ATTEST_OVERRIDE_ARGUMENT}
    --expected-revision "${TXNOPT_BUILD_GIT_REVISION}"
    --expected-tree "${TXNOPT_BUILD_GIT_TREE}"
    --expected-manifest-sha256 "${TXNOPT_BUILD_SOURCE_MANIFEST_SHA256}"
    --expected-source-dirty "${TXNOPT_BUILD_SOURCE_DIRTY_TEXT}"
    --expected-development-override "${TXNOPT_BUILD_DEVELOPMENT_OVERRIDE_TEXT}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Verifying native source-byte attestation"
  VERBATIM
)
if(NOT TXNOPT_ALLOW_DIRTY_BUILD)
  add_custom_command(
    TARGET native_source_attestation_check POST_BUILD
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
      verify-materialized-cpp
      --root "${CMAKE_CURRENT_SOURCE_DIR}"
      --revision "${TXNOPT_BUILD_GIT_REVISION}"
      --destination "${TXNOPT_NATIVE_SOURCE_SNAPSHOT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Verifying commit-materialized native C++ sources"
    VERBATIM
  )
endif()
add_dependencies(_native native_source_attestation_check)

function(txnopt_add_post_build_attestation target_name)
  add_custom_command(
    TARGET ${target_name} POST_BUILD
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
      verify
      --root "${CMAKE_CURRENT_SOURCE_DIR}"
      ${TXNOPT_ATTEST_OVERRIDE_ARGUMENT}
      --expected-revision "${TXNOPT_BUILD_GIT_REVISION}"
      --expected-tree "${TXNOPT_BUILD_GIT_TREE}"
      --expected-manifest-sha256 "${TXNOPT_BUILD_SOURCE_MANIFEST_SHA256}"
      --expected-source-dirty "${TXNOPT_BUILD_SOURCE_DIRTY_TEXT}"
      --expected-development-override "${TXNOPT_BUILD_DEVELOPMENT_OVERRIDE_TEXT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Rechecking native source-byte attestation after linking"
    VERBATIM
  )
  if(NOT TXNOPT_ALLOW_DIRTY_BUILD)
    add_custom_command(
      TARGET ${target_name} POST_BUILD
      COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/native_build_attestation.py"
        verify-materialized-cpp
        --root "${CMAKE_CURRENT_SOURCE_DIR}"
        --revision "${TXNOPT_BUILD_GIT_REVISION}"
        --destination "${TXNOPT_NATIVE_SOURCE_SNAPSHOT}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      COMMENT "Rechecking commit-materialized C++ sources after linking"
      VERBATIM
    )
  endif()
endfunction()
txnopt_add_post_build_attestation(_native)

set(TXNOPT_BUILD_ATTESTATION_DEFINITIONS
  TXNOPT_BUILD_GIT_REVISION="${TXNOPT_BUILD_GIT_REVISION}"
  TXNOPT_BUILD_GIT_TREE="${TXNOPT_BUILD_GIT_TREE}"
  TXNOPT_BUILD_SOURCE_MANIFEST_SHA256="${TXNOPT_BUILD_SOURCE_MANIFEST_SHA256}"
  TXNOPT_BUILD_TRACKED_FILE_COUNT=${TXNOPT_BUILD_TRACKED_FILE_COUNT}
  TXNOPT_BUILD_SOURCE_DIRTY=${TXNOPT_BUILD_SOURCE_DIRTY}
  TXNOPT_BUILD_DEVELOPMENT_OVERRIDE=${TXNOPT_BUILD_DEVELOPMENT_OVERRIDE}
  TXNOPT_BUILD_CPP_SOURCE_KIND="${TXNOPT_BUILD_CPP_SOURCE_KIND}"
  TXNOPT_BUILD_PERFORMANCE_PROFILE="${TXNOPT_PERFORMANCE_PROFILE}"
  TXNOPT_BUILD_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  TXNOPT_BUILD_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  TXNOPT_BUILD_INTERPROCEDURAL_OPTIMIZATION=${TXNOPT_BUILD_INTERPROCEDURAL_OPTIMIZATION}
  TXNOPT_BUILD_HOST_NATIVE=${TXNOPT_BUILD_HOST_NATIVE}
)
target_compile_definitions(_native PRIVATE ${TXNOPT_BUILD_ATTESTATION_DEFINITIONS})

if(TXNOPT_BUILD_INTERPROCEDURAL_OPTIMIZATION)
  set_property(TARGET _native PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(MSVC)
  target_compile_options(_native PRIVATE /O2)
else()
  target_compile_options(_native PRIVATE -O3 -Wall -Wextra -Wpedantic)
  if(TXNOPT_BUILD_HOST_NATIVE)
    target_compile_options(_native PRIVATE -march=native)
  endif()
endif()

if(NOT TXNOPT_SANITIZER STREQUAL "none")
  if(MSVC)
    message(FATAL_ERROR "TXNOPT_SANITIZER profiles require GCC or Clang")
  elseif(
    NOT TXNOPT_SANITIZER STREQUAL "address-undefined"
    AND NOT TXNOPT_SANITIZER STREQUAL "thread"
  )
    message(
      FATAL_ERROR
      "TXNOPT_SANITIZER must be none, address-undefined, or thread"
    )
  endif()

  add_executable(
    txnopt_native_concurrency_sanitizer
    "${TXNOPT_NATIVE_CPP_DIR}/txnopt_core/concurrency_sanitizer.cpp"
  )
  target_include_directories(
    txnopt_native_concurrency_sanitizer PRIVATE "${TXNOPT_NATIVE_CPP_DIR}"
  )
  target_compile_options(
    txnopt_native_concurrency_sanitizer PRIVATE -O1 -g -fno-omit-frame-pointer
  )
  foreach(_TXNOPT_SANITIZER_TARGET _native txnopt_native_concurrency_sanitizer)
    if(TXNOPT_SANITIZER STREQUAL "address-undefined")
      target_compile_options(
        ${_TXNOPT_SANITIZER_TARGET} PRIVATE -fsanitize=address,undefined
      )
      target_link_options(
        ${_TXNOPT_SANITIZER_TARGET} PRIVATE -fsanitize=address,undefined
      )
    else()
      target_compile_options(${_TXNOPT_SANITIZER_TARGET} PRIVATE -fsanitize=thread)
      target_link_options(${_TXNOPT_SANITIZER_TARGET} PRIVATE -fsanitize=thread)
    endif()
  endforeach()
  enable_testing()
  if(
    TXNOPT_SANITIZER STREQUAL "thread"
    AND UNIX
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"
  )
    find_program(TXNOPT_SETARCH_EXECUTABLE setarch REQUIRED)
    add_test(
      NAME txnopt-native-concurrency-sanitizer
      COMMAND
        "${TXNOPT_SETARCH_EXECUTABLE}" "${CMAKE_SYSTEM_PROCESSOR}" -R
        "$<TARGET_FILE:txnopt_native_concurrency_sanitizer>"
    )
  else()
    add_test(
      NAME txnopt-native-concurrency-sanitizer
      COMMAND txnopt_native_concurrency_sanitizer
    )
  endif()
endif()

install(TARGETS _native DESTINATION txnopt)
