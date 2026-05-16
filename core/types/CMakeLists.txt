add_library(clob_types INTERFACE)
target_include_directories(clob_types INTERFACE ${CMAKE_SOURCE_DIR})
target_compile_features(clob_types INTERFACE cxx_std_23)
add_library(clob::types ALIAS clob_types)
