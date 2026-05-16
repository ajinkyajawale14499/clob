#include "core/orderbook/book.hpp"

#include <catch2/catch_test_macros.hpp>

using clob::Book;
using clob::Order;
using clob::OrderId;
using clob::Price;
using clob::Quantity;
using clob::Side;

TEST_CASE("Book: empty by default", "[book]") {
    Book b;
    REQUIRE_FALSE(b.best_bid().has_value());
    REQUIRE_FALSE(b.best_ask().has_value());
}

TEST_CASE("Book: add_limit places bid at correct level", "[book]") {
    Book b;
    b.add_limit(Price{10000}, Order{OrderId{1}, Side::Bid, Quantity{5}});
    REQUIRE(b.best_bid() == Price{10000});
}

TEST_CASE("Book: bid_top < ask_top after adds on opposite sides", "[book]") {
    Book b;
    b.add_limit(Price{9900},  Order{OrderId{1}, Side::Bid, Quantity{5}});
    b.add_limit(Price{10100}, Order{OrderId{2}, Side::Ask, Quantity{5}});
    REQUIRE(b.best_bid() < b.best_ask());
}

TEST_CASE("Book: cancel removes order and updates index", "[book]") {
    Book b;
    b.add_limit(Price{10000}, Order{OrderId{1}, Side::Bid, Quantity{5}});
    REQUIRE(b.cancel(OrderId{1}));
    REQUIRE_FALSE(b.best_bid().has_value());
    REQUIRE_FALSE(b.find(OrderId{1}).has_value());
}

TEST_CASE("Book: cancel of unknown id returns false", "[book]") {
    Book b;
    REQUIRE_FALSE(b.cancel(OrderId{999}));
}

TEST_CASE("Book: unindex removes index entry without touching level", "[book]") {
    Book b;
    b.add_limit(Price{10000}, Order{OrderId{1}, Side::Bid, Quantity{5}});
    b.unindex(OrderId{1});
    // Order still in level — but the matching engine calls unindex after
    // consume_front fully consumes a maker (so the level itself is also empty).
    REQUIRE_FALSE(b.find(OrderId{1}).has_value());
}
