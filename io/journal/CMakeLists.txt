# Header-only for W6.1 — writer/reader add .cpp files in W6.2/W6.3.
add_library(clob_journal INTERFACE)
target_include_directories(clob_journal INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(clob_journal INTERFACE clob::types)
target_compile_features(clob_journal INTERFACE cxx_std_23)
add_library(clob::journal ALIAS clob_journal)
