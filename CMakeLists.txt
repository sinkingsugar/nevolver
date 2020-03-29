project("chainblocks-tgbot")
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -march=sandybridge -ffast-math)

#### Files to format and tidy
set(
  MY_PROJECT_SOURCE_FILES
  ${MY_PROJECT_SOURCE_FILES}
  ${CMAKE_CURRENT_LIST_DIR}/nevolver.hpp
  ${CMAKE_CURRENT_LIST_DIR}/tests/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/bench1.cpp
  ${CMAKE_CURRENT_LIST_DIR}/random.hpp
  ${CMAKE_CURRENT_LIST_DIR}/squash.hpp
  ${CMAKE_CURRENT_LIST_DIR}/connections.hpp
  ${CMAKE_CURRENT_LIST_DIR}/node.hpp
  ${CMAKE_CURRENT_LIST_DIR}/nodes/annhidden.hpp
  ${CMAKE_CURRENT_LIST_DIR}/network.hpp
  ${CMAKE_CURRENT_LIST_DIR}/networks/perceptron.hpp
  ${CMAKE_CURRENT_LIST_DIR}/networks/narx.hpp
  ${CMAKE_CURRENT_LIST_DIR}/networks/lstm.hpp
  )

#### Header paths for tidy
set(
  MY_PROJECT_HEADER_PATHS
  ${MY_PROJECT_HEADER_PATHS}
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

if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
endif()

if(APPLE)
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)
endif()

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/deps/cereal/include
  ${CMAKE_CURRENT_LIST_DIR}/deps/easyloggingpp/src
  ${CMAKE_CURRENT_LIST_DIR}/deps/Catch2/single_include
  )

add_executable(
  nevolver
  ${CMAKE_CURRENT_LIST_DIR}/deps/easyloggingpp/src/easylogging++.cc
  ${CMAKE_CURRENT_LIST_DIR}/tests/main.cpp
  )
