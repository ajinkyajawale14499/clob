#pragma once

#include <variant>

#include "core/types/order_id.hpp"
#include "core/types/price.hpp"
#include "core/types/quantity.hpp"
#include "core/types/side.hpp"

namespace clob {

struct NewLimit {
    OrderId id;
    Side side;
    Price price;
    Quantity qty;
    auto operator<=>(const NewLimit&) const noexcept = default;
};

struct NewMarket {
    OrderId id;
    Side side;
    Quantity qty;
    auto operator<=>(const NewMarket&) const noexcept = default;
};

struct NewIoc {
    OrderId id;
    Side side;
    Price price;
    Quantity qty;
    auto operator<=>(const NewIoc&) const noexcept = default;
};

struct Cancel {
    OrderId id;
    auto operator<=>(const Cancel&) const noexcept = default;
};

struct Replace {
    OrderId old_id;
    OrderId new_id;
    Price price;
    Quantity qty;
    auto operator<=>(const Replace&) const noexcept = default;
};

using OrderEvent = std::variant<NewLimit, NewMarket, NewIoc, Cancel, Replace>;

}  // namespace clob
