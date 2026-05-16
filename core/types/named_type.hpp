#pragma once

#include <compare>

namespace clob {

// Minimal strong-type wrapper. Tag a primitive with a Tag to get a distinct type.
// Usage: struct PriceTag {}; using Price = NamedType<int64_t, PriceTag>;
template <typename T, typename Tag>
class NamedType {
public:
    using underlying_type = T;

    constexpr NamedType() noexcept = default;
    constexpr explicit NamedType(T value) noexcept : value_{value} {}

    [[nodiscard]] constexpr T value() const noexcept { return value_; }

    constexpr auto operator<=>(const NamedType&) const noexcept = default;

private:
    T value_{};
};

}  // namespace clob
