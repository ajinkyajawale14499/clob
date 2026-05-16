#include "core/orderbook/level.hpp"
#include "core/orderbook/order.hpp"

#include <catch2/catch_test_macros.hpp>

using clob::Level;
using clob::Order;
using clob::OrderId;
using clob::Quantity;
using clob::Side;

TEST_CASE("Level: empty by default", "[level]") {
    Level lvl;
    REQUIRE(lvl.empty());
    REQUIRE(lvl.total_quantity() == Quantity{0});
}

TEST_CASE("Level: add preserves FIFO", "[level]") {
    Level lvl;
    lvl.add(Order{OrderId{1}, Side::Bid, Quantity{10}});
    lvl.add(Order{OrderId{2}, Side::Bid, Quantity{20}});
    REQUIRE(lvl.total_quantity() == Quantity{30});
    REQUIRE(lvl.front().id == OrderId{1});
}

TEST_CASE("Level: erase_by_id returns true on hit, false on miss", "[level]") {
    Level lvl;
    lvl.add(Order{OrderId{1}, Side::Bid, Quantity{10}});
    REQUIRE(lvl.erase_by_id(OrderId{1}));
    REQUIRE(lvl.empty());
    REQUIRE_FALSE(lvl.erase_by_id(OrderId{1}));  // already gone
}

TEST_CASE("Level: consume_front returns id of any fully-filled head", "[level]") {
    Level lvl;
    lvl.add(Order{OrderId{1}, Side::Bid, Quantity{10}});
    // Take less than head qty -> no full-consumption signal.
    REQUIRE_FALSE(lvl.consume_front(Quantity{5}).has_value());
    REQUIRE(lvl.front().quantity == Quantity{5});
    // Take the rest -> returns id, level becomes empty.
    auto consumed = lvl.consume_front(Quantity{5});
    REQUIRE(consumed.has_value());
    REQUIRE(*consumed == OrderId{1});
    REQUIRE(lvl.empty());
}
