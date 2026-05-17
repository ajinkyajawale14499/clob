cmake_minimum_required(VERSION 3.28)
project(clob LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build all static libs as PIC. Required because bindings/clob_py is a shared
# module (.so) that links the matching/scoring static libs; under gcc-13 + GNU
# ld + ASan, the ASan TLS-style symbols can't be relocated into a shared
# object unless every input was compiled -fPIC. Apple Clang handles this
# silently which is why the local macOS build never tripped on it.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CLOB_WARN_AS_ERROR "Treat warnings as errors" ON)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if (CLOB_WARN_AS_ERROR)
        add_compile_options(-Werror)
    endif()
endif()

find_package(Catch2 3 REQUIRED)
include(CTest)
include(Catch)

add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(apps)
add_subdirectory(benchmarks)

option(CLOB_BUILD_PYTHON "Build the clob_py pybind11 module" ON)
if (CLOB_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
