#include "core/types/order_id.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("OrderId: construct + compare", "[order_id]") {
    using clob::OrderId;
    REQUIRE(OrderId{1} != OrderId{2});
    REQUIRE(OrderId{1} < OrderId{2});
}
