#include "core/types/quantity.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Quantity: construct + compare", "[quantity]") {
    using clob::Quantity;
    Quantity q{42};
    REQUIRE(q.value() == 42);
    REQUIRE(q < Quantity{100});
}
