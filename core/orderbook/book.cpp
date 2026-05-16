#include "core/orderbook/book.hpp"

namespace clob {

std::optional<Price> Book::best_bid() const noexcept {
    return bids_.empty() ? std::nullopt : std::optional{bids_.begin()->first};
}

std::optional<Price> Book::best_ask() const noexcept {
    return asks_.empty() ? std::nullopt : std::optional{asks_.begin()->first};
}

void Book::add_limit(Price price, Order order) {
    if (order.side == Side::Bid) {
        bids_[price].add(order);
    } else {
        asks_[price].add(order);
    }
    id_index_[order.id.value()] = Location{order.side, price};
}

bool Book::cancel(OrderId id) {
    auto it = id_index_.find(id.value());
    if (it == id_index_.end()) return false;
    Location loc = it->second;
    id_index_.erase(it);

    bool erased = false;
    if (loc.side == Side::Bid) {
        auto lit = bids_.find(loc.price);
        if (lit != bids_.end()) {
            erased = lit->second.erase_by_id(id);
            if (lit->second.empty()) bids_.erase(lit);
        }
    } else {
        auto lit = asks_.find(loc.price);
        if (lit != asks_.end()) {
            erased = lit->second.erase_by_id(id);
            if (lit->second.empty()) asks_.erase(lit);
        }
    }
    return erased;
}

std::optional<Book::Location> Book::find(OrderId id) const noexcept {
    auto it = id_index_.find(id.value());
    return it == id_index_.end() ? std::nullopt : std::optional{it->second};
}

void Book::unindex(OrderId id) noexcept {
    id_index_.erase(id.value());
}

Level* Book::level_at(Side side, Price price) {
    if (side == Side::Bid) {
        auto it = bids_.find(price);
        return it == bids_.end() ? nullptr : &it->second;
    } else {
        auto it = asks_.find(price);
        return it == asks_.end() ? nullptr : &it->second;
    }
}

const Level* Book::level_at(Side side, Price price) const {
    if (side == Side::Bid) {
        auto it = bids_.find(price);
        return it == bids_.end() ? nullptr : &it->second;
    } else {
        auto it = asks_.find(price);
        return it == asks_.end() ? nullptr : &it->second;
    }
}

void Book::drop_if_empty(Side side, Price price) {
    if (side == Side::Bid) {
        auto it = bids_.find(price);
        if (it != bids_.end() && it->second.empty()) bids_.erase(it);
    } else {
        auto it = asks_.find(price);
        if (it != asks_.end() && it->second.empty()) asks_.erase(it);
    }
}

bool Book::empty(Side side) const noexcept {
    return side == Side::Bid ? bids_.empty() : asks_.empty();
}

}  // namespace clob
