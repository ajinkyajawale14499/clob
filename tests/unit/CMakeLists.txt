add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_smoke DISCOVERY_MODE PRE_TEST)

foreach(name price quantity order_id level book engine engine_journal)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE clob::matching Catch2::Catch2WithMain)
    catch_discover_tests(test_${name} DISCOVERY_MODE PRE_TEST)
endforeach()

add_executable(test_feature_state test_feature_state.cpp)
target_link_libraries(test_feature_state PRIVATE clob::scoring Catch2::Catch2WithMain)
catch_discover_tests(test_feature_state DISCOVERY_MODE PRE_TEST)

foreach(name journal_event journal_writer journal_reader fill_log)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE clob::journal Catch2::Catch2WithMain)
    catch_discover_tests(test_${name} DISCOVERY_MODE PRE_TEST)
endforeach()
