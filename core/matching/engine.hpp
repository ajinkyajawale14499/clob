#pragma once

#include <vector>

#include "core/matching/fill.hpp"
#include "core/orderbook/book.hpp"

namespace clob {

class Engine {
public:
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
    Book book_;
};

}  // namespace clob
