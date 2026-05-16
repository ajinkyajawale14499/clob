#include "core/matching/engine.hpp"

#include <algorithm>  // std::min
#include <optional>

namespace clob {

namespace {

// Match `qty` of taker against opposite side, optionally capped at price_cap.
// price_cap = nullopt -> market order (walk to empty book).
std::vector<Fill> match_against(Book& book,
                                OrderId taker,
                                Side taker_side,
                                std::optional<Price> price_cap,
                                Quantity qty) {
    std::vector<Fill> fills;
    Quantity remaining = qty;
    const Side opp = opposite(taker_side);

    auto top_opp = [&]() -> std::optional<Price> {
        return taker_side == Side::Bid ? book.best_ask() : book.best_bid();
    };

    auto in_cap = [&](Price p) -> bool {
        if (!price_cap) return true;
        return taker_side == Side::Bid ? p.value() <= price_cap->value()
                                       : p.value() >= price_cap->value();
    };

    while (remaining.value() > 0) {
        auto top = top_opp();
        if (!top || !in_cap(*top)) break;
        const Price match_price = *top;

        for (Level* lvl = book.level_at(opp, match_price);
             lvl != nullptr && remaining.value() > 0 && !lvl->empty();
             lvl = book.level_at(opp, match_price)) {
            const Order head_copy = lvl->front();
            const Quantity take{std::min(remaining.value(), head_copy.quantity.value())};
            fills.push_back({taker, head_copy.id, match_price, take});
            remaining = Quantity{remaining.value() - take.value()};

            if (auto consumed = lvl->consume_front(take); consumed) {
                book.unindex(*consumed);
            }
        }
        book.drop_if_empty(opp, match_price);
    }
    return fills;
}

}  // namespace

std::vector<Fill> Engine::add_limit(OrderId id, Side side, Price price, Quantity qty) {
    if (qty.value() <= 0) return {};
    auto fills = match_against(book_, id, side, price, qty);
    std::int64_t consumed = 0;
    for (const auto& f : fills) consumed += f.quantity.value();
    const Quantity remaining{qty.value() - consumed};
    if (remaining.value() > 0) {
        // Reject duplicate resting OrderId (see Task 3.4 rationale).
        if (book_.find(id).has_value()) return fills;
        book_.add_limit(price, Order{id, side, remaining});
    }
    return fills;
}

std::vector<Fill> Engine::add_market(OrderId id, Side side, Quantity qty) {
    if (qty.value() <= 0) return {};
    return match_against(book_, id, side, std::nullopt, qty);
}

}  // namespace clob
