#include "io/journal/fill_log.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

using clob::Fill;
using clob::FillLogReader;
using clob::FillLogWriter;
using clob::OrderId;
using clob::Price;
using clob::Quantity;

namespace {

std::filesystem::path tmp_path(const char* suffix) {
    return std::filesystem::temp_directory_path() /
           (std::string("clob_fill_log_") + suffix + ".bin");
}

std::vector<unsigned char> read_all(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST_CASE("FillLogWriter: each record is exactly 32 bytes", "[fill_log]") {
    auto path = tmp_path("size");
    std::filesystem::remove(path);
    {
        FillLogWriter w(path);
        w.write(Fill{OrderId{1}, OrderId{2}, Price{100}, Quantity{5}});
        w.write(Fill{OrderId{3}, OrderId{4}, Price{101}, Quantity{6}});
    }
    auto bytes = read_all(path);
    REQUIRE(bytes.size() == 64);  // 2 records x 32 bytes
    std::filesystem::remove(path);
}

TEST_CASE("FillLogReader: round-trips Fills", "[fill_log]") {
    auto path = tmp_path("rt");
    std::filesystem::remove(path);

    const Fill fills[] = {
        {OrderId{1}, OrderId{2}, Price{100}, Quantity{5}},
        {OrderId{3}, OrderId{4}, Price{101}, Quantity{6}},
        {OrderId{5}, OrderId{6}, Price{102}, Quantity{7}},
    };
    {
        FillLogWriter w(path);
        for (const auto& f : fills) w.write(f);
    }

    FillLogReader r(path);
    for (const auto& expected : fills) {
        auto got = r.next();
        REQUIRE(got.has_value());
        REQUIRE(got->taker_id == expected.taker_id);
        REQUIRE(got->maker_id == expected.maker_id);
        REQUIRE(got->price == expected.price);
        REQUIRE(got->quantity == expected.quantity);
    }
    REQUIRE_FALSE(r.next().has_value());
    REQUIRE(r.end_reason() == FillLogReader::EndReason::CleanEof);

    std::filesystem::remove(path);
}

TEST_CASE("FillLogReader: trailing partial record is TruncatedTrailingRecord",
          "[fill_log][recovery]") {
    auto path = tmp_path("trunc");
    std::filesystem::remove(path);

    {
        FillLogWriter w(path);
        w.write(Fill{OrderId{1}, OrderId{2}, Price{100}, Quantity{5}});
    }
    // Append 10 bytes of partial record (< 32).
    {
        std::ofstream f(path, std::ios::binary | std::ios::app);
        const char garbage[10] = {};
        f.write(garbage, 10);
    }

    FillLogReader r(path);
    REQUIRE(r.next().has_value());            // first record intact
    REQUIRE_FALSE(r.next().has_value());      // partial -> nullopt
    REQUIRE(r.end_reason() == FillLogReader::EndReason::TruncatedTrailingRecord);

    std::filesystem::remove(path);
}
