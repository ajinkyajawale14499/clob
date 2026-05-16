// Determinism integration test (ADR 0001).
//
// Strategy:
//   1. Run a fixed event sequence through Engine WITH JournalWriter sink ->
//      a journal file PLUS an in-memory baseline fill vector.
//   2. Serialize the baseline fills to an expected fill log file.
//   3. Run replay-cli on the journal twice, into two distinct output files.
//   4. Assert: replay_A == replay_B (determinism — same input twice, same bytes)
//      AND     replay_A == expected (replay correctness — replay reproduces
//      the original execution).

#include "core/matching/engine.hpp"
#include "io/journal/fill_log.hpp"
#include "io/journal/journal_writer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace clob;

namespace {

std::vector<unsigned char> read_all(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Drive a fixed scenario. Touches every code path: add_limit (rest + cross),
// add_market, add_ioc, cancel, cancel_replace.
std::vector<Fill> run_scenario_inline(Engine& e) {
    std::vector<Fill> out;
    auto run = [&](auto fills) { for (const auto& f : fills) out.push_back(f); };

    // Build a 2-sided book.
    run(e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5}));
    run(e.add_limit(OrderId{2}, Side::Bid, Price{99},  Quantity{8}));
    run(e.add_limit(OrderId{3}, Side::Ask, Price{102}, Quantity{6}));
    run(e.add_limit(OrderId{4}, Side::Ask, Price{103}, Quantity{4}));

    // Aggressive limit that walks the ask side partially.
    run(e.add_limit(OrderId{5}, Side::Bid, Price{102}, Quantity{4}));

    // Market that finishes the first ask level and walks into the next.
    run(e.add_market(OrderId{6}, Side::Bid, Quantity{5}));

    // IOC at a price that doesn't cross — drops with no fills.
    run(e.add_ioc(OrderId{7}, Side::Bid, Price{50}, Quantity{3}));

    // Cancel an existing resting bid.
    e.cancel(OrderId{2});

    // Cancel-replace: move bid 1 to a new id at a new price.
    run(e.cancel_replace(OrderId{1}, OrderId{8}, Price{99}, Quantity{5}));

    // Another aggressive limit to ensure replace target works.
    run(e.add_limit(OrderId{9}, Side::Ask, Price{99}, Quantity{3}));

    return out;
}

}  // namespace

TEST_CASE("Replay determinism: journal -> replay-cli twice -> byte-identical fills",
          "[determinism][integration]") {
    auto tmp_dir = std::filesystem::temp_directory_path() / "clob_determinism";
    std::filesystem::create_directories(tmp_dir);

    const auto journal_path  = tmp_dir / "scenario.journal.bin";
    const auto expected_path = tmp_dir / "expected.fills.bin";
    const auto replay_a_path = tmp_dir / "replay_a.fills.bin";
    const auto replay_b_path = tmp_dir / "replay_b.fills.bin";
    for (const auto& p : {journal_path, expected_path, replay_a_path, replay_b_path}) {
        std::filesystem::remove(p);
    }

    // (1) Capture journal + inline baseline.
    std::vector<Fill> baseline;
    {
        JournalWriter w(journal_path);
        Engine e([&](const OrderEvent& ev) { w.write(ev); });
        baseline = run_scenario_inline(e);
    }

    // (2) Write baseline as a fill log.
    {
        FillLogWriter w(expected_path);
        for (const auto& f : baseline) w.write(f);
    }

    // (3) Run replay-cli twice. CLOB_REPLAY_CLI is injected via CMake.
    auto run_replay = [&](const std::filesystem::path& out) {
        std::string cmd = std::string(CLOB_REPLAY_CLI) + " " +
                          journal_path.string() + " " + out.string() +
                          " > /dev/null";
        const int rc = std::system(cmd.c_str());
        REQUIRE(rc == 0);
    };
    run_replay(replay_a_path);
    run_replay(replay_b_path);

    // (4) Two replays must be byte-identical to each other AND to baseline.
    const auto a = read_all(replay_a_path);
    const auto b = read_all(replay_b_path);
    const auto expected = read_all(expected_path);

    REQUIRE(a == b);          // determinism: same input twice -> same bytes
    REQUIRE(a == expected);   // correctness: replay reproduces original
    REQUIRE(a.size() % 32 == 0);
    REQUIRE_FALSE(a.empty());

    // Clean up.
    for (const auto& p : {journal_path, expected_path, replay_a_path, replay_b_path}) {
        std::filesystem::remove(p);
    }
}
