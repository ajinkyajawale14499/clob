#include <catch2/catch_test_macros.hpp>

TEST_CASE("toolchain smoke: arithmetic", "[smoke]") {
    REQUIRE(2 + 2 == 4);
}
