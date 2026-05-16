add_library(clob_matching STATIC engine.cpp)
target_include_directories(clob_matching PUBLIC ${CMAKE_SOURCE_DIR})
# clob::scoring is in core/ (not io/), so this is convention-compliant.
target_link_libraries(clob_matching PUBLIC clob::orderbook clob::events clob::scoring)
target_compile_features(clob_matching PUBLIC cxx_std_23)
add_library(clob::matching ALIAS clob_matching)
