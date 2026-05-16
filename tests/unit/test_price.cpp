#include "core/types/price.hpp"

#include <catch2/catch_test_macros.hpp>

using clob::Price;

TEST_CASE("Price: construction and value access", "[price]") {
    Price p{12345};
    REQUIRE(p.value() == 12345);
}

TEST_CASE("Price: comparison operators", "[price]") {
    Price a{100};
    Price b{200};
    REQUIRE(a < b);
    REQUIRE(b > a);
    REQUIRE(a != b);
    REQUIRE(a == Price{100});
}
