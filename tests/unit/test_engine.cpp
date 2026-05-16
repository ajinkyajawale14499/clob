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

TEST_CASE("Engine: market order fills against best opposite", "[engine][market]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Ask, Price{100}, Quantity{5});
    e.add_limit(OrderId{2}, Side::Ask, Price{101}, Quantity{5});
    auto fills = e.add_market(OrderId{3}, Side::Bid, Quantity{7});
    REQUIRE(fills.size() == 2);
    REQUIRE(fills[0].price == Price{100});
    REQUIRE(fills[0].quantity == Quantity{5});
    REQUIRE(fills[1].price == Price{101});
    REQUIRE(fills[1].quantity == Quantity{2});
}

TEST_CASE("Engine: market stops at empty book (no rest)", "[engine][market]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Ask, Price{100}, Quantity{5});
    auto fills = e.add_market(OrderId{2}, Side::Bid, Quantity{10});
    REQUIRE(fills.size() == 1);
    REQUIRE_FALSE(e.book().best_bid().has_value());
}

TEST_CASE("Engine: IOC fills what crosses, cancels remainder", "[engine][ioc]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Ask, Price{100}, Quantity{3});
    auto fills = e.add_ioc(OrderId{2}, Side::Bid, Price{100}, Quantity{10});
    REQUIRE(fills.size() == 1);
    REQUIRE(fills[0].quantity == Quantity{3});
    REQUIRE_FALSE(e.book().best_bid().has_value());  // 7 dropped, not rested
}

TEST_CASE("Engine: cancel removes resting order", "[engine][cancel]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});
    REQUIRE(e.cancel(OrderId{1}));
    REQUIRE_FALSE(e.book().best_bid().has_value());
}

TEST_CASE("Engine: cancel unknown returns false", "[engine][cancel]") {
    Engine e;
    REQUIRE_FALSE(e.cancel(OrderId{999}));
}

TEST_CASE("Engine: cancel_replace loses time priority", "[engine][replace]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});
    e.add_limit(OrderId{2}, Side::Bid, Price{100}, Quantity{5});
    e.cancel_replace(OrderId{1}, OrderId{3}, Price{100}, Quantity{5});
    const auto* lvl = e.book().level_at(Side::Bid, Price{100});  // const overload
    REQUIRE(lvl->front().id == OrderId{2});
}

TEST_CASE("Engine: cancel_replace with duplicate new_id rejects atomically (old preserved)",
          "[engine][replace][edge]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});
    e.add_limit(OrderId{2}, Side::Bid, Price{99}, Quantity{3});
    // Try to replace 1 with 2 — but 2 is already resting -> atomic reject.
    auto fills = e.cancel_replace(OrderId{1}, OrderId{2}, Price{100}, Quantity{5});
    REQUIRE(fills.empty());
    REQUIRE(e.book().find(OrderId{1}).has_value());  // old NOT destroyed
}

TEST_CASE("Engine: cancel_replace with qty=0 rejects atomically (old preserved)",
          "[engine][replace][edge]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});
    auto fills = e.cancel_replace(OrderId{1}, OrderId{2}, Price{100}, Quantity{0});
    REQUIRE(fills.empty());
    REQUIRE(e.book().find(OrderId{1}).has_value());
}

TEST_CASE("Engine: self-trade silently fills (STP not implemented in v1)",
          "[engine][known_limitation]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});
    // Same id, opposite side, crossing — matcher allows it. Pins current behavior.
    auto fills = e.add_limit(OrderId{1}, Side::Ask, Price{100}, Quantity{5});
    REQUIRE(fills.size() == 1);
    REQUIRE(fills[0].maker_id == fills[0].taker_id);
}

TEST_CASE("Engine: qty=1 round-trip", "[engine][edge]") {
    Engine e;
    e.add_limit(OrderId{1}, Side::Ask, Price{100}, Quantity{1});
    auto fills = e.add_limit(OrderId{2}, Side::Bid, Price{100}, Quantity{1});
    REQUIRE(fills.size() == 1);
    REQUIRE(fills[0].quantity == Quantity{1});
    REQUIRE_FALSE(e.book().best_ask().has_value());
}
