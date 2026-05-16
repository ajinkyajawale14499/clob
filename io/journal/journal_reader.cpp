#include "io/journal/journal_reader.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace clob {

namespace {

constexpr std::uint8_t TAG_NEW_LIMIT  = 0x01;
constexpr std::uint8_t TAG_NEW_MARKET = 0x02;
constexpr std::uint8_t TAG_NEW_IOC    = 0x03;
constexpr std::uint8_t TAG_CANCEL     = 0x04;
constexpr std::uint8_t TAG_REPLACE    = 0x05;

constexpr std::uint32_t MAX_RECORD_BYTES = 1024;  // sanity cap

template <typename T>
T read_le(const std::uint8_t* p) {
    static_assert(std::endian::native == std::endian::little);
    std::array<std::uint8_t, sizeof(T)> buf{};
    for (std::size_t i = 0; i < sizeof(T); ++i) buf[i] = p[i];
    return std::bit_cast<T>(buf);
}

}  // namespace

JournalReader::JournalReader(const std::filesystem::path& path)
    : stream_(path, std::ios::binary) {
    if (!stream_) {
        throw std::runtime_error("JournalReader: cannot open " + path.string());
    }
}

std::optional<OrderEvent> JournalReader::next() {
    // Read 4-byte length prefix.
    std::array<std::uint8_t, 4> length_buf{};
    stream_.read(reinterpret_cast<char*>(length_buf.data()), 4);
    const auto got_len_bytes = stream_.gcount();
    if (got_len_bytes == 0 && stream_.eof()) {
        end_reason_ = EndReason::CleanEof;
        return std::nullopt;
    }
    if (got_len_bytes != 4) {
        end_reason_ = EndReason::TruncatedTrailingRecord;
        return std::nullopt;
    }

    const std::uint32_t length = read_le<std::uint32_t>(length_buf.data());
    if (length == 0 || length > MAX_RECORD_BYTES) {
        throw std::runtime_error("JournalReader: implausible record length " +
                                 std::to_string(length));
    }

    // Read full payload.
    std::vector<std::uint8_t> payload(length);
    stream_.read(reinterpret_cast<char*>(payload.data()), length);
    if (stream_.gcount() != static_cast<std::streamsize>(length)) {
        end_reason_ = EndReason::TruncatedTrailingRecord;
        return std::nullopt;
    }

    const std::uint8_t tag = payload[0];
    const std::uint8_t* p = payload.data() + 1;

    auto read_id = [&]() -> OrderId {
        OrderId v{read_le<std::uint64_t>(p)};
        p += sizeof(std::uint64_t);
        return v;
    };
    auto read_price = [&]() -> Price {
        Price v{read_le<std::int64_t>(p)};
        p += sizeof(std::int64_t);
        return v;
    };
    auto read_qty = [&]() -> Quantity {
        Quantity v{read_le<std::int64_t>(p)};
        p += sizeof(std::int64_t);
        return v;
    };
    auto read_side = [&]() -> Side {
        Side s = static_cast<Side>(static_cast<std::int8_t>(*p));
        ++p;
        return s;
    };

    switch (tag) {
        case TAG_NEW_LIMIT: {
            if (length != 1 + 8 + 1 + 8 + 8) break;  // fall through to error
            auto id = read_id();
            auto side = read_side();
            auto price = read_price();
            auto qty = read_qty();
            return OrderEvent{NewLimit{id, side, price, qty}};
        }
        case TAG_NEW_MARKET: {
            if (length != 1 + 8 + 1 + 8) break;
            auto id = read_id();
            auto side = read_side();
            auto qty = read_qty();
            return OrderEvent{NewMarket{id, side, qty}};
        }
        case TAG_NEW_IOC: {
            if (length != 1 + 8 + 1 + 8 + 8) break;
            auto id = read_id();
            auto side = read_side();
            auto price = read_price();
            auto qty = read_qty();
            return OrderEvent{NewIoc{id, side, price, qty}};
        }
        case TAG_CANCEL: {
            if (length != 1 + 8) break;
            auto id = read_id();
            return OrderEvent{Cancel{id}};
        }
        case TAG_REPLACE: {
            if (length != 1 + 8 + 8 + 8 + 8) break;
            auto old_id = read_id();
            auto new_id = read_id();
            auto price = read_price();
            auto qty = read_qty();
            return OrderEvent{Replace{old_id, new_id, price, qty}};
        }
        default:
            throw std::runtime_error("JournalReader: unknown tag " +
                                     std::to_string(tag));
    }
    throw std::runtime_error("JournalReader: length/tag mismatch (length=" +
                             std::to_string(length) + ", tag=" +
                             std::to_string(tag) + ")");
}

}  // namespace clob
