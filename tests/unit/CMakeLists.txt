add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_smoke DISCOVERY_MODE PRE_TEST)

foreach(name price quantity order_id)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE clob::types Catch2::Catch2WithMain)
    catch_discover_tests(test_${name} DISCOVERY_MODE PRE_TEST)
endforeach()
