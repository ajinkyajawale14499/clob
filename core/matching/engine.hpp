#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "core/events/order_event.hpp"
#include "core/matching/fill.hpp"
#include "core/orderbook/book.hpp"

namespace clob {

class Engine {
public:
    // Optional sink invoked before each accepted mutating operation. W6 wires
    // this to JournalWriter so journals replay bit-identically. Default is
    // empty -> no journaling (existing tests rely on this).
    using JournalSink = std::function<void(const OrderEvent&)>;

    Engine() = default;
    explicit Engine(JournalSink sink) noexcept : journal_sink_(std::move(sink)) {}

    std::vector<Fill> add_limit(OrderId id, Side side, Price price, Quantity qty);
    std::vector<Fill> add_market(OrderId id, Side side, Quantity qty);
    std::vector<Fill> add_ioc(OrderId id, Side side, Price price, Quantity qty);

    bool cancel(OrderId id);
    std::vector<Fill> cancel_replace(OrderId old_id,
                                     OrderId new_id,
                                     Price price,
                                     Quantity qty);

    [[nodiscard]] const Book& book() const noexcept { return book_; }

private:
    // Inner helper used both by public add_limit and by cancel_replace. The
    // public path fires the sink before calling this; cancel_replace fires only
    // its own Replace event so the journal stays normalized (one record per
    // top-level operation, no nested events).
    std::vector<Fill> do_add_limit(OrderId id, Side side, Price price, Quantity qty);

    Book book_;
    JournalSink journal_sink_;
};

}  // namespace clob
