#include "io/journal/journal_event.hpp"

#include <catch2/catch_test_macros.hpp>

#include <variant>

using clob::Cancel;
using clob::NewIoc;
using clob::NewLimit;
using clob::NewMarket;
using clob::OrderEvent;
using clob::OrderId;
using clob::Price;
using clob::Quantity;
using clob::Replace;
using clob::Side;

TEST_CASE("OrderEvent: holds NewLimit", "[journal_event]") {
    OrderEvent ev = NewLimit{OrderId{1}, Side::Bid, Price{100}, Quantity{5}};
    REQUIRE(std::holds_alternative<NewLimit>(ev));
    const auto& nl = std::get<NewLimit>(ev);
    REQUIRE(nl.id == OrderId{1});
    REQUIRE(nl.side == Side::Bid);
    REQUIRE(nl.price == Price{100});
    REQUIRE(nl.qty == Quantity{5});
}

TEST_CASE("OrderEvent: holds NewMarket", "[journal_event]") {
    OrderEvent ev = NewMarket{OrderId{2}, Side::Ask, Quantity{10}};
    REQUIRE(std::holds_alternative<NewMarket>(ev));
    REQUIRE(std::get<NewMarket>(ev).qty == Quantity{10});
}

TEST_CASE("OrderEvent: holds NewIoc", "[journal_event]") {
    OrderEvent ev = NewIoc{OrderId{3}, Side::Bid, Price{101}, Quantity{7}};
    REQUIRE(std::holds_alternative<NewIoc>(ev));
    REQUIRE(std::get<NewIoc>(ev).price == Price{101});
}

TEST_CASE("OrderEvent: holds Cancel", "[journal_event]") {
    OrderEvent ev = Cancel{OrderId{4}};
    REQUIRE(std::holds_alternative<Cancel>(ev));
    REQUIRE(std::get<Cancel>(ev).id == OrderId{4});
}

TEST_CASE("OrderEvent: holds Replace", "[journal_event]") {
    OrderEvent ev = Replace{OrderId{5}, OrderId{6}, Price{102}, Quantity{3}};
    REQUIRE(std::holds_alternative<Replace>(ev));
    REQUIRE(std::get<Replace>(ev).old_id == OrderId{5});
    REQUIRE(std::get<Replace>(ev).new_id == OrderId{6});
}

TEST_CASE("OrderEvent: equality", "[journal_event]") {
    OrderEvent a = NewLimit{OrderId{1}, Side::Bid, Price{100}, Quantity{5}};
    OrderEvent b = NewLimit{OrderId{1}, Side::Bid, Price{100}, Quantity{5}};
    OrderEvent c = NewLimit{OrderId{1}, Side::Bid, Price{100}, Quantity{6}};
    REQUIRE(a == b);
    REQUIRE_FALSE(a == c);
}
