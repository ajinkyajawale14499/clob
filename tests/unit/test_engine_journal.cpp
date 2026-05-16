#include "core/matching/engine.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using clob::Cancel;
using clob::Engine;
using clob::NewIoc;
using clob::NewLimit;
using clob::NewMarket;
using clob::OrderEvent;
using clob::OrderId;
using clob::Price;
using clob::Quantity;
using clob::Replace;
using clob::Side;

TEST_CASE("Engine: default ctor has no sink — existing behavior unchanged",
          "[engine][journal]") {
    Engine e;  // no sink
    auto fills = e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});
    REQUIRE(fills.empty());
    REQUIRE(e.book().best_bid() == Price{100});
}

TEST_CASE("Engine: sink captures NewLimit before mutation", "[engine][journal]") {
    std::vector<OrderEvent> captured;
    Engine e([&](const OrderEvent& ev) { captured.push_back(ev); });

    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});

    REQUIRE(captured.size() == 1);
    REQUIRE(captured[0] == OrderEvent{NewLimit{OrderId{1}, Side::Bid, Price{100}, Quantity{5}}});
}

TEST_CASE("Engine: sink captures NewMarket and NewIoc", "[engine][journal]") {
    std::vector<OrderEvent> captured;
    Engine e([&](const OrderEvent& ev) { captured.push_back(ev); });

    // Seed an ask so market/ioc have something to hit.
    e.add_limit(OrderId{1}, Side::Ask, Price{100}, Quantity{10});
    e.add_market(OrderId{2}, Side::Bid, Quantity{3});
    e.add_ioc(OrderId{3}, Side::Bid, Price{100}, Quantity{2});

    REQUIRE(captured.size() == 3);
    REQUIRE(captured[1] == OrderEvent{NewMarket{OrderId{2}, Side::Bid, Quantity{3}}});
    REQUIRE(captured[2] == OrderEvent{NewIoc{OrderId{3}, Side::Bid, Price{100}, Quantity{2}}});
}

TEST_CASE("Engine: zero-qty is rejected pre-sink — NOT journaled", "[engine][journal]") {
    std::vector<OrderEvent> captured;
    Engine e([&](const OrderEvent& ev) { captured.push_back(ev); });
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{0});
    REQUIRE(captured.empty());
}

TEST_CASE("Engine: cancel of unknown id NOT journaled", "[engine][journal]") {
    std::vector<OrderEvent> captured;
    Engine e([&](const OrderEvent& ev) { captured.push_back(ev); });
    REQUIRE_FALSE(e.cancel(OrderId{999}));
    REQUIRE(captured.empty());
}

TEST_CASE("Engine: successful cancel is journaled exactly once", "[engine][journal]") {
    std::vector<OrderEvent> captured;
    Engine e([&](const OrderEvent& ev) { captured.push_back(ev); });
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});  // captured[0]
    REQUIRE(e.cancel(OrderId{1}));                                // captured[1]
    REQUIRE(captured.size() == 2);
    REQUIRE(captured[1] == OrderEvent{Cancel{OrderId{1}}});
}

TEST_CASE("Engine: cancel_replace fires Replace once (no nested NewLimit)",
          "[engine][journal]") {
    std::vector<OrderEvent> captured;
    Engine e([&](const OrderEvent& ev) { captured.push_back(ev); });
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});       // captured[0]
    e.cancel_replace(OrderId{1}, OrderId{2}, Price{99}, Quantity{6});  // captured[1]

    REQUIRE(captured.size() == 2);
    REQUIRE(captured[1] ==
            OrderEvent{Replace{OrderId{1}, OrderId{2}, Price{99}, Quantity{6}}});
}

TEST_CASE("Engine: rejected cancel_replace is NOT journaled", "[engine][journal]") {
    std::vector<OrderEvent> captured;
    Engine e([&](const OrderEvent& ev) { captured.push_back(ev); });
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});  // captured[0]

    auto fills = e.cancel_replace(OrderId{999}, OrderId{2}, Price{99}, Quantity{6});
    REQUIRE(fills.empty());
    REQUIRE(captured.size() == 1);  // no Replace journaled
}
