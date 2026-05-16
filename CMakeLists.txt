cmake_minimum_required(VERSION 3.28)
project(clob LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
