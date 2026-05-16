#pragma once

#include "core/types/order_id.hpp"
#include "core/types/quantity.hpp"
#include "core/types/side.hpp"

namespace clob {

struct Order {
    OrderId id;
    Side side;
    Quantity quantity;
};

}  // namespace clob
