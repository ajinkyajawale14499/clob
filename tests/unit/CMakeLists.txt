add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_smoke DISCOVERY_MODE PRE_TEST)
