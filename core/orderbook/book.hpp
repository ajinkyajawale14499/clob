#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <unordered_map>

#include "core/orderbook/level.hpp"
#include "core/types/price.hpp"
#include "core/types/side.hpp"

namespace clob {

class Book {
public:
    struct Location {
        Side side;
        Price price;
    };

    [[nodiscard]] std::optional<Price> best_bid() const noexcept;
    [[nodiscard]] std::optional<Price> best_ask() const noexcept;

    void add_limit(Price price, Order order);

    // Cancel = remove from level + drop empty level + remove from index.
    bool cancel(OrderId id);

    // Find location without removing. Used by Engine for cancel_replace.
    [[nodiscard]] std::optional<Location> find(OrderId id) const noexcept;

    // Remove only the index entry (used by matching engine after fills consume an order).
    void unindex(OrderId id) noexcept;

    // Access level for matching engine (mutable) and for read-only inspection (const).
    Level* level_at(Side side, Price price);
    [[nodiscard]] const Level* level_at(Side side, Price price) const;
    void drop_if_empty(Side side, Price price);

    [[nodiscard]] bool empty(Side side) const noexcept;

private:
    std::map<Price, Level, std::greater<>> bids_;  // descending: begin() = highest
    std::map<Price, Level> asks_;                  // ascending:  begin() = lowest

    std::unordered_map<std::uint64_t, Location> id_index_;
};

}  // namespace clob
