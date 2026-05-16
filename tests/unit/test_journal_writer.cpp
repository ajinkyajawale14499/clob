#include "io/journal/journal_writer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

using clob::JournalWriter;
using clob::NewLimit;
using clob::OrderEvent;
using clob::OrderId;
using clob::Price;
using clob::Quantity;
using clob::Side;

namespace {

std::filesystem::path tmp_path(const char* suffix) {
    return std::filesystem::temp_directory_path() /
           (std::string("clob_journal_test_") + suffix + ".bin");
}

std::vector<unsigned char> read_all_bytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST_CASE("JournalWriter: writes one NewLimit with correct byte layout", "[journal_writer]") {
    auto path = tmp_path("one_newlimit");
    std::filesystem::remove(path);  // clean start

    {
        JournalWriter w(path);
        w.write(NewLimit{OrderId{0x0102030405060708ULL}, Side::Bid,
                         Price{0x1011121314151617LL}, Quantity{0x2021222324252627LL}});
    }  // dtor closes

    auto bytes = read_all_bytes(path);
    // 4-byte length + 1-byte tag + 25 payload = 30 bytes total.
    REQUIRE(bytes.size() == 30);
    // length field: little-endian uint32 = 26 (tag + payload)
    REQUIRE(bytes[0] == 26);
    REQUIRE(bytes[1] == 0);
    REQUIRE(bytes[2] == 0);
    REQUIRE(bytes[3] == 0);
    // tag byte: 0x01 = NewLimit
    REQUIRE(bytes[4] == 0x01);
    // first 8 payload bytes: OrderId little-endian
    REQUIRE(bytes[5] == 0x08);
    REQUIRE(bytes[6] == 0x07);
    REQUIRE(bytes[12] == 0x01);  // high byte of id

    std::filesystem::remove(path);
}

TEST_CASE("JournalWriter: appending writes multiple records", "[journal_writer]") {
    auto path = tmp_path("multi");
    std::filesystem::remove(path);

    {
        JournalWriter w(path);
        w.write(NewLimit{OrderId{1}, Side::Bid, Price{100}, Quantity{5}});
        w.write(NewLimit{OrderId{2}, Side::Ask, Price{101}, Quantity{6}});
    }

    auto bytes = read_all_bytes(path);
    REQUIRE(bytes.size() == 60);

    std::filesystem::remove(path);
}

TEST_CASE("JournalWriter: reopening appends, does not truncate", "[journal_writer]") {
    auto path = tmp_path("reopen");
    std::filesystem::remove(path);

    {
        JournalWriter w(path);
        w.write(NewLimit{OrderId{1}, Side::Bid, Price{100}, Quantity{5}});
    }
    {
        JournalWriter w(path);  // re-open
        w.write(NewLimit{OrderId{2}, Side::Ask, Price{200}, Quantity{6}});
    }

    auto bytes = read_all_bytes(path);
    REQUIRE(bytes.size() == 60);  // both records present

    std::filesystem::remove(path);
}
