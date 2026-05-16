#pragma once

#include <cstdint>

namespace clob {

enum class Side : std::int8_t { Bid = 1, Ask = -1 };

constexpr Side opposite(Side s) noexcept {
    return s == Side::Bid ? Side::Ask : Side::Bid;
}

}  // namespace clob
