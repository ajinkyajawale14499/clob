#pragma once

#include <cstdint>

#include "core/types/named_type.hpp"

namespace clob {

struct QuantityTag {};
using Quantity = NamedType<std::int64_t, QuantityTag>;

}  // namespace clob
