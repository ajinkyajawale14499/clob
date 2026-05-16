#pragma once

#include "core/types/order_id.hpp"
#include "core/types/price.hpp"
#include "core/types/quantity.hpp"

namespace clob {

struct Fill {
    OrderId taker_id;
    OrderId maker_id;
    Price price;
    Quantity quantity;
};

}  // namespace clob
