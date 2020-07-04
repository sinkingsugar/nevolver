project("chainblocks-tgbot")
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -march=sandybridge -ffast-math -Wno-multichar)

add_compile_definitions(ELPP_THREAD_SAFE)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(ELPP_DISABLE_TRACE_LOGS)
  add_compile_definitions(ELPP_DISABLE_DEBUG_LOGS)
else()
  add_compile_definitions(ELPP_DISABLE_TRACE_LOGS)
  add_compile_options(-Wall)
endif()

# remember CMAKE_EXPORT_COMPILE_COMMANDS , for irony etc

### CLANG SANITIZERS (mac / linux only)
# Build Types
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}
  CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel tsan asan lsan msan ubsan"
  FORCE)

# ThreadSanitizer
set(CMAKE_C_FLAGS_TSAN
  "-fsanitize=thread -g -O1"
  CACHE STRING "Flags used by the C compiler during ThreadSanitizer builds."
  FORCE)
set(CMAKE_CXX_FLAGS_TSAN
  "-DCB_USE_TSAN -fsanitize=thread -g -O1"
  CACHE STRING "Flags used by the C++ compiler during ThreadSanitizer builds."
  FORCE)

# AddressSanitize
set(CMAKE_C_FLAGS_ASAN
  "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O1"
  CACHE STRING "Flags used by the C compiler during AddressSanitizer builds."
  FORCE)
set(CMAKE_CXX_FLAGS_ASAN
  "-DBOOST_USE_ASAN -fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -O1"
  CACHE STRING "Flags used by the C++ compiler during AddressSanitizer builds."
  FORCE)

# LeakSanitizer
set(CMAKE_C_FLAGS_LSAN
  "-fsanitize=leak -fno-omit-frame-pointer -g -O1"
  CACHE STRING "Flags used by the C compiler during LeakSanitizer builds."
  FORCE)
set(CMAKE_CXX_FLAGS_LSAN
  "-fsanitize=leak -fno-omit-frame-pointer -g -O1"
  CACHE STRING "Flags used by the C++ compiler during LeakSanitizer builds."
  FORCE)

# MemorySanitizer
set(CMAKE_C_FLAGS_MSAN
  "-fsanitize=memory -fno-optimize-sibling-calls -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -g -O2"
  CACHE STRING "Flags used by the C compiler during MemorySanitizer builds."
  FORCE)
set(CMAKE_CXX_FLAGS_MSAN
  "-fsanitize=memory -fno-optimize-sibling-calls -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer -g -O2"
  CACHE STRING "Flags used by the C++ compiler during MemorySanitizer builds."
  FORCE)

# UndefinedBehaviour
set(CMAKE_C_FLAGS_UBSAN
  "-fsanitize=undefined"
  CACHE STRING "Flags used by the C compiler during UndefinedBehaviourSanitizer builds."
  FORCE)
set(CMAKE_CXX_FLAGS_UBSAN
  "-fsanitize=undefined"
  CACHE STRING "Flags used by the C++ compiler during UndefinedBehaviourSanitizer builds."
  FORCE)
### SANITIZERS

if (NOT DEFINED CHAINBLOCKS_DIR)
  set(CHAINBLOCKS_DIR "${CMAKE_CURRENT_LIST_DIR}/deps/chainblocks")
endif()

#### Files to format and tidy
set(
  MY_PROJECT_SOURCE_FILES
  ${MY_PROJECT_SOURCE_FILES}
  ${CMAKE_CURRENT_LIST_DIR}/nevolver.hpp
  ${CMAKE_CURRENT_LIST_DIR}/neurofloat.hpp
  ${CMAKE_CURRENT_LIST_DIR}/tests/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/squash.hpp
  ${CMAKE_CURRENT_LIST_DIR}/connections.hpp
  ${CMAKE_CURRENT_LIST_DIR}/node.hpp
  ${CMAKE_CURRENT_LIST_DIR}/nodes/annhidden.hpp
  ${CMAKE_CURRENT_LIST_DIR}/network.hpp
  ${CMAKE_CURRENT_LIST_DIR}/networks/mlp.hpp
  ${CMAKE_CURRENT_LIST_DIR}/networks/narx.hpp
  ${CMAKE_CURRENT_LIST_DIR}/networks/lstm.hpp
  ${CMAKE_CURRENT_LIST_DIR}/networks/liquid.hpp
  ${CMAKE_CURRENT_LIST_DIR}/chainblocks/blocks.cpp
  )

#### Header paths for tidy
set(
  MY_PROJECT_HEADER_PATHS
  ${MY_PROJECT_HEADER_PATHS}
  -I${CMAKE_CURRENT_LIST_DIR}
  -I${CHAINBLOCKS_DIR}/include
  -I${CHAINBLOCKS_DIR}/deps/magic_enum/include
  -I${CHAINBLOCKS_DIR}/deps/nameof/include
  -I${CHAINBLOCKS_DIR}/deps/cpp-taskflow
  -I${CMAKE_CURRENT_LIST_DIR}/deps/cereal/include
  -I${CMAKE_CURRENT_LIST_DIR}/deps/easyloggingpp/src
  -I${CMAKE_CURRENT_LIST_DIR}/deps/Catch2/single_include
  )

### setup clang format
find_program(
  CLANG_FORMAT_EXE
  NAMES "clang-format"
  DOC "Path to clang-format executable"
  )
if(NOT CLANG_FORMAT_EXE)
  message(STATUS "clang-format not found.")
else()
  message(STATUS "clang-format found: ${CLANG_FORMAT_EXE}")
endif()

#### Format target
if(CLANG_FORMAT_EXE)
  add_custom_target(
    format
    COMMAND ${CLANG_FORMAT_EXE} -i -style=LLVM ${MY_PROJECT_SOURCE_FILES}
    )
endif()
###

### setup clang tidy
find_program(
  CLANG_TIDY_EXE
  NAMES "clang-tidy"
  DOC "Path to clang-tidy executable"
  )
if(NOT CLANG_TIDY_EXE)
  message(STATUS "clang-tidy not found.")
else()
  message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
endif()

#### tidy target
set(tidyfied)
foreach(_file ${MY_PROJECT_SOURCE_FILES})
  if(CLANG_TIDY_EXE)
    add_custom_command(
      OUTPUT ${_file}.noop
      COMMAND ${CLANG_TIDY_EXE} -checks=-*,clang-analyzer-*,performance-*,bugprone-* -fix ${_file} -- -std=c++17 -DDEBUG ${MY_PROJECT_HEADER_PATHS}
      )
    list(APPEND tidyfied ${_file}.noop)
  endif()
endforeach()
add_custom_target(tidy DEPENDS ${tidyfied})
###

### general
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  if(WIN32)
    # add_link_options(-fuse-ld=lld)
    # add_compile_options(-gcodeview -Wl,-pdb=)
  endif()
elseif(WIN32)
  add_link_options(-static)
endif()

if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
endif()

if(APPLE)
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)
endif()

include_directories(
  ${CHAINBLOCKS_DIR}/include
  ${CHAINBLOCKS_DIR}/deps/cpp-taskflow
  ${CHAINBLOCKS_DIR}/deps/magic_enum/include
  ${CHAINBLOCKS_DIR}/deps/nameof/include
  ${CMAKE_CURRENT_LIST_DIR}/deps/cereal/include
  ${CMAKE_CURRENT_LIST_DIR}/deps/easyloggingpp/src
  ${CMAKE_CURRENT_LIST_DIR}/deps/Catch2/single_include
  )

add_executable(
  nevolver
  ${CMAKE_CURRENT_LIST_DIR}/deps/easyloggingpp/src/easylogging++.cc
  ${CMAKE_CURRENT_LIST_DIR}/tests/main.cpp
  )

add_library(cbnevolver SHARED
  ${CMAKE_CURRENT_LIST_DIR}/deps/easyloggingpp/src/easylogging++.cc
  ${CMAKE_CURRENT_LIST_DIR}/chainblocks/blocks.cpp)

target_compile_options(cbnevolver PUBLIC
  -fvisibility=hidden
  )

set_target_properties(cbnevolver PROPERTIES PREFIX "")
set_target_properties(cbnevolver PROPERTIES OUTPUT_NAME "nevolver")
