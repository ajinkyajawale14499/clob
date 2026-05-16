#pragma once

#include <cstdint>

#include "core/types/named_type.hpp"

namespace clob {

struct OrderIdTag {};
using OrderId = NamedType<std::uint64_t, OrderIdTag>;

}  // namespace clob
