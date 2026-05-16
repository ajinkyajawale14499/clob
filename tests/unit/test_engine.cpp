#include "core/matching/engine.hpp"

#include <catch2/catch_test_macros.hpp>

using clob::Engine;
using clob::OrderId;
using clob::Price;
using clob::Quantity;
using clob::Side;

TEST_CASE("Engine: passive limit rests when no cross", "[engine]") {
    Engine e;
    auto fills = e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{10});
    REQUIRE(fills.empty());
    REQUIRE(e.book().best_bid() == Price{100});
}

TEST_CASE("Engine: aggressive limit fills + maker fully consumed leaves no index", "[engine]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Ask, Price{100}, Quantity{6});
    auto fills = e.add_limit(OrderId{2}, Side::Bid, Price{100}, Quantity{6});
    REQUIRE(fills.size() == 1);
    REQUIRE(fills[0].maker_id == OrderId{1});
    REQUIRE(fills[0].taker_id == OrderId{2});
    REQUIRE(fills[0].quantity == Quantity{6});
    // Maker fully consumed — index should be clean.
    REQUIRE_FALSE(e.book().find(OrderId{1}).has_value());
}

TEST_CASE("Engine: limit walks multiple levels", "[engine]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Ask, Price{100}, Quantity{3});
    e.add_limit(OrderId{2}, Side::Ask, Price{101}, Quantity{5});
    e.add_limit(OrderId{3}, Side::Ask, Price{102}, Quantity{10});
    auto fills = e.add_limit(OrderId{4}, Side::Bid, Price{102}, Quantity{12});
    REQUIRE(fills.size() == 3);
    REQUIRE(fills[0].price == Price{100});
    REQUIRE(fills[1].price == Price{101});
    REQUIRE(fills[2].price == Price{102});
    REQUIRE(fills[2].quantity == Quantity{4});
}

TEST_CASE("Engine: limit rests remainder after partial walk", "[engine]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Ask, Price{100}, Quantity{3});
    auto fills = e.add_limit(OrderId{2}, Side::Bid, Price{100}, Quantity{10});
    REQUIRE(fills.size() == 1);
    REQUIRE(e.book().best_bid() == Price{100});
}

TEST_CASE("Engine: zero-quantity order is rejected (no fill, no rest)", "[engine][edge]") {
    Engine e;
    auto fills = e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{0});
    REQUIRE(fills.empty());
    REQUIRE_FALSE(e.book().best_bid().has_value());
}

TEST_CASE("Engine: duplicate resting OrderId is rejected (no rest, original intact)",
          "[engine][edge]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});
    // Re-using id 1 on a non-crossing limit: matcher rejects the rest, original survives.
    auto fills = e.add_limit(OrderId{1}, Side::Bid, Price{99}, Quantity{3});
    REQUIRE(fills.empty());
    REQUIRE(e.book().best_bid() == Price{100});  // original 100-level still there
}
