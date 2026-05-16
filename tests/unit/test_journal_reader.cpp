#include "io/journal/journal_reader.hpp"
#include "io/journal/journal_writer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

using clob::Cancel;
using clob::JournalReader;
using clob::JournalWriter;
using clob::NewIoc;
using clob::NewLimit;
using clob::NewMarket;
using clob::OrderEvent;
using clob::OrderId;
using clob::Price;
using clob::Quantity;
using clob::Replace;
using clob::Side;

namespace {

std::filesystem::path tmp_path(const char* suffix) {
    return std::filesystem::temp_directory_path() /
           (std::string("clob_journal_reader_") + suffix + ".bin");
}

}  // namespace

TEST_CASE("JournalReader: empty file yields nothing, clean EOF", "[journal_reader]") {
    auto path = tmp_path("empty");
    std::filesystem::remove(path);
    { std::ofstream f(path, std::ios::binary); }  // create empty

    JournalReader r(path);
    REQUIRE_FALSE(r.next().has_value());
    REQUIRE(r.end_reason() == JournalReader::EndReason::CleanEof);

    std::filesystem::remove(path);
}

TEST_CASE("JournalReader: round-trips all 5 event kinds", "[journal_reader]") {
    auto path = tmp_path("all_kinds");
    std::filesystem::remove(path);

    const OrderEvent events[] = {
        NewLimit{OrderId{1}, Side::Bid, Price{100}, Quantity{5}},
        NewMarket{OrderId{2}, Side::Ask, Quantity{6}},
        NewIoc{OrderId{3}, Side::Bid, Price{99}, Quantity{7}},
        Cancel{OrderId{1}},
        Replace{OrderId{4}, OrderId{5}, Price{101}, Quantity{8}},
    };

    {
        JournalWriter w(path);
        for (const auto& e : events) w.write(e);
    }

    JournalReader r(path);
    for (const auto& expected : events) {
        auto got = r.next();
        REQUIRE(got.has_value());
        REQUIRE(*got == expected);
    }
    REQUIRE_FALSE(r.next().has_value());
    REQUIRE(r.end_reason() == JournalReader::EndReason::CleanEof);

    std::filesystem::remove(path);
}

TEST_CASE("JournalReader: truncated length prefix reports TruncatedTrailingRecord",
          "[journal_reader][recovery]") {
    auto path = tmp_path("trunc_len");
    std::filesystem::remove(path);

    {
        JournalWriter w(path);
        w.write(NewLimit{OrderId{1}, Side::Bid, Price{100}, Quantity{5}});
    }
    // Append 2 bytes — partial length prefix (simulates kill-9 mid-write).
    {
        std::ofstream f(path, std::ios::binary | std::ios::app);
        const char garbage[] = {0x1A, 0x00};
        f.write(garbage, 2);
    }

    JournalReader r(path);
    auto first = r.next();
    REQUIRE(first.has_value());  // first record intact
    REQUIRE_FALSE(r.next().has_value());
    REQUIRE(r.end_reason() == JournalReader::EndReason::TruncatedTrailingRecord);

    std::filesystem::remove(path);
}

TEST_CASE("JournalReader: truncated payload reports TruncatedTrailingRecord",
          "[journal_reader][recovery]") {
    auto path = tmp_path("trunc_payload");
    std::filesystem::remove(path);

    {
        JournalWriter w(path);
        w.write(NewLimit{OrderId{1}, Side::Bid, Price{100}, Quantity{5}});
    }
    // Append full 4-byte length declaring a 25-byte payload, then only 3 bytes.
    {
        std::ofstream f(path, std::ios::binary | std::ios::app);
        const char hdr[] = {0x1A, 0x00, 0x00, 0x00};  // length = 26
        const char partial_payload[] = {0x01, 0x02, 0x03};
        f.write(hdr, 4);
        f.write(partial_payload, 3);
    }

    JournalReader r(path);
    REQUIRE(r.next().has_value());  // good record
    REQUIRE_FALSE(r.next().has_value());
    REQUIRE(r.end_reason() == JournalReader::EndReason::TruncatedTrailingRecord);

    std::filesystem::remove(path);
}
