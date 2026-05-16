#pragma once

#include <vector>

#include "core/matching/fill.hpp"
#include "core/orderbook/book.hpp"

namespace clob {

class Engine {
public:
    std::vector<Fill> add_limit(OrderId id, Side side, Price price, Quantity qty);
    std::vector<Fill> add_market(OrderId id, Side side, Quantity qty);

    [[nodiscard]] const Book& book() const noexcept { return book_; }

private:
    Book book_;
};

}  // namespace clob
