find_package(rapidcheck REQUIRED)

add_executable(test_book_invariants test_book_invariants.cpp)
target_link_libraries(test_book_invariants
    PRIVATE clob::matching Catch2::Catch2WithMain rapidcheck::rapidcheck
)
# rapidcheck uses std::aligned_storage (deprecated in C++23) and depends on
# <exception> being included before its own headers. Suppress the deprecation
# locally — our own code stays under -Werror.
target_compile_options(test_book_invariants PRIVATE
    -Wno-deprecated-declarations
)
catch_discover_tests(test_book_invariants DISCOVERY_MODE PRE_TEST)
