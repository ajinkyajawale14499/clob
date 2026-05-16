add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(test_smoke DISCOVERY_MODE PRE_TEST)

foreach(name price quantity order_id level book engine)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE clob::matching Catch2::Catch2WithMain)
    catch_discover_tests(test_${name} DISCOVERY_MODE PRE_TEST)
endforeach()

add_executable(test_journal_event test_journal_event.cpp)
target_link_libraries(test_journal_event PRIVATE clob::journal Catch2::Catch2WithMain)
catch_discover_tests(test_journal_event DISCOVERY_MODE PRE_TEST)
