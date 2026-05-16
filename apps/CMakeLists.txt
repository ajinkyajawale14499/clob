add_executable(matcher-cli matcher-main.cpp)
target_link_libraries(matcher-cli PRIVATE clob::matching)
