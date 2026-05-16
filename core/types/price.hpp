#pragma once

#include <cstdint>

#include "core/types/named_type.hpp"

namespace clob {

struct PriceTag {};

// Price in ticks (int64). LOBSTER convention: 1 tick = 1/10^4 dollars.
// Binance: 1 tick = 1/10^8 USDT. Document the source convention per ingestion path.
using Price = NamedType<std::int64_t, PriceTag>;

}  // namespace clob
