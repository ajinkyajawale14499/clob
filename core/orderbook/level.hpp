#pragma once

#include <cstddef>
#include <deque>
#include <optional>

#include "core/orderbook/order.hpp"

namespace clob {

class Level {
public:
    [[nodiscard]] bool empty() const noexcept { return orders_.empty(); }
    [[nodiscard]] Quantity total_quantity() const noexcept { return total_; }
    [[nodiscard]] const Order& front() const noexcept { return orders_.front(); }
    [[nodiscard]] std::size_t size() const noexcept { return orders_.size(); }

    void add(Order o) noexcept {
        total_ = Quantity{total_.value() + o.quantity.value()};
        orders_.push_back(o);
    }

    // Consume up to `qty` from the head. Returns the OrderId IF the head was
    // fully consumed (so callers can update any external index atomically).
    std::optional<OrderId> consume_front(Quantity qty) noexcept {
        Order& head = orders_.front();
        if (qty.value() < head.quantity.value()) {
            head.quantity = Quantity{head.quantity.value() - qty.value()};
            total_ = Quantity{total_.value() - qty.value()};
            return std::nullopt;
        }
        // Full consumption.
        OrderId consumed_id = head.id;
        total_ = Quantity{total_.value() - head.quantity.value()};
        orders_.pop_front();
        return consumed_id;
    }

    bool erase_by_id(OrderId id) noexcept {
        for (auto it = orders_.begin(); it != orders_.end(); ++it) {
            if (it->id == id) {
                total_ = Quantity{total_.value() - it->quantity.value()};
                orders_.erase(it);
                return true;
            }
        }
        return false;
    }

    // Used by property tests to verify index <-> level consistency. Const-friendly.
    [[nodiscard]] bool contains(OrderId id) const noexcept {
        for (const auto& o : orders_) {
            if (o.id == id) return true;
        }
        return false;
    }

private:
    std::deque<Order> orders_;
    Quantity total_{};
};

}  // namespace clob
