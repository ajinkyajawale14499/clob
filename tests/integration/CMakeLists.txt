add_executable(test_determinism test_determinism.cpp)
target_link_libraries(test_determinism
    PRIVATE clob::journal Catch2::Catch2WithMain
)
# Inject the replay-cli path so the test can shell out to it.
target_compile_definitions(test_determinism PRIVATE
    CLOB_REPLAY_CLI="$<TARGET_FILE:replay-cli>"
)
add_dependencies(test_determinism replay-cli)
catch_discover_tests(test_determinism DISCOVERY_MODE PRE_TEST)
