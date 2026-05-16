add_executable(matcher-cli matcher-main.cpp)
target_link_libraries(matcher-cli PRIVATE clob::journal)

add_executable(replay-cli replay-main.cpp)
# clob::journal transitively links clob::matching — list it once to avoid
# duplicate-library warnings from the linker.
target_link_libraries(replay-cli PRIVATE clob::journal)
