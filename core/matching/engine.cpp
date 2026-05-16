#include "core/matching/engine.hpp"

#include <algorithm>  // std::min
#include <optional>   // std::optional (used by match_against in Task 3.5)

namespace clob {

std::vector<Fill> Engine::add_limit(OrderId id, Side side, Price price, Quantity qty) {
    std::vector<Fill> fills;
    if (qty.value() <= 0) return fills;  // zero/negative rejected silently

    Quantity remaining = qty;
    const Side opp = opposite(side);

    auto crosses = [&]() -> std::optional<Price> {
        if (side == Side::Bid) {
            auto ba = book_.best_ask();
            return (ba && price.value() >= ba->value()) ? ba : std::nullopt;
        } else {
            auto bb = book_.best_bid();
            return (bb && price.value() <= bb->value()) ? bb : std::nullopt;
        }
    };

    while (remaining.value() > 0) {
        auto cross = crosses();
        if (!cross) break;
        const Price match_price = *cross;

        // Drain inner loop. consume_front returns the maker id if fully consumed.
        for (Level* lvl = book_.level_at(opp, match_price);
             lvl != nullptr && remaining.value() > 0 && !lvl->empty();
             lvl = book_.level_at(opp, match_price)) {
            const Order head_copy = lvl->front();  // copy id+qty BEFORE mutating
            const Quantity take{std::min(remaining.value(), head_copy.quantity.value())};
            fills.push_back({id, head_copy.id, match_price, take});
            remaining = Quantity{remaining.value() - take.value()};

            if (auto consumed = lvl->consume_front(take); consumed) {
                book_.unindex(*consumed);  // keep id_index_ consistent on full fill
            }
        }

        // Cleanup empty level — re-fetch fresh, never via stale lvl pointer.
        book_.drop_if_empty(opp, match_price);
    }

    if (remaining.value() > 0) {
        // Reject duplicate resting OrderId — otherwise original orphaned in its level.
        if (book_.find(id).has_value()) {
            return fills;  // No rest. Caller can detect dup via fills.empty() + book unchanged.
        }
        book_.add_limit(price, Order{id, side, remaining});
    }
    return fills;
}

}  // namespace clob
