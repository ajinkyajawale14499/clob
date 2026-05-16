#include "core/matching/engine.hpp"
#include "core/scoring/scorer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <optional>
#include <vector>

using namespace clob;

namespace {

std::filesystem::path repo_root() {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
}

bool artifacts_present() {
    const auto root = repo_root();
    return std::filesystem::exists(root / "model" / "artifacts" / "model.onnx") &&
           std::filesystem::exists(root / "model" / "artifacts" / "microprice_g.json");
}

}  // namespace

TEST_CASE("Engine default ctor: no scoring path", "[engine][scoring]") {
    // No model required — scoring is opt-in.
    int score_calls = 0;
    Engine e(
        /*journal_sink=*/nullptr,
        /*scorer=*/nullptr,
        /*score_sink=*/[&](OrderId, double) { ++score_calls; },
        /*ticker=*/"AAPL",
        /*lut=*/nullptr);
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});
    REQUIRE(score_calls == 0);  // no scorer -> sink never fires
}

TEST_CASE("Engine with scoring: sink fires on every accepted op", "[engine][scoring]") {
    if (!artifacts_present()) {
        SKIP("model.onnx + microprice_g.json not present");
    }
    Scorer scorer(repo_root() / "model" / "artifacts" / "model.onnx");
    auto lut = MicropriceLut::load(repo_root() / "model" / "artifacts" /
                                    "microprice_g.json");

    std::vector<std::pair<std::uint64_t, double>> captured;
    Engine e(nullptr, &scorer,
              [&](OrderId id, double s) { captured.emplace_back(id.value(), s); },
              "AAPL", &lut);

    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{5});
    e.add_limit(OrderId{2}, Side::Ask, Price{101}, Quantity{5});
    e.cancel(OrderId{1});

    REQUIRE(captured.size() == 3);
    REQUIRE(captured[0].first == 1);
    REQUIRE(captured[1].first == 2);
    REQUIRE(captured[2].first == 1);  // cancel id
    // Every score in [-1, +1].
    for (auto [id, score] : captured) {
        REQUIRE(score >= -1.0);
        REQUIRE(score <= 1.0);
    }
}

TEST_CASE("Engine with scoring: rejected ops do NOT fire sink",
          "[engine][scoring][edge]") {
    if (!artifacts_present()) {
        SKIP("artifacts not present");
    }
    Scorer scorer(repo_root() / "model" / "artifacts" / "model.onnx");
    auto lut = MicropriceLut::load(repo_root() / "model" / "artifacts" /
                                    "microprice_g.json");

    int score_calls = 0;
    Engine e(nullptr, &scorer,
              [&](OrderId, double) { ++score_calls; }, "AAPL", &lut);

    // qty=0 -> rejected pre-sink
    e.add_limit(OrderId{1}, Side::Bid, Price{100}, Quantity{0});
    REQUIRE(score_calls == 0);

    // unknown cancel -> no state change, no sink fire
    REQUIRE_FALSE(e.cancel(OrderId{999}));
    REQUIRE(score_calls == 0);
}

TEST_CASE("Engine: fills are byte-identical with/without scorer (observer-only)",
          "[engine][scoring][determinism]") {
    if (!artifacts_present()) {
        SKIP("artifacts not present");
    }

    // Run scenario through Engine WITHOUT scorer.
    Engine plain;
    std::vector<Fill> plain_fills;
    auto run = [&](Engine& engine) -> std::vector<Fill> {
        std::vector<Fill> all;
        auto extend = [&](std::vector<Fill> v) {
            for (auto& f : v) all.push_back(f);
        };
        extend(engine.add_limit(OrderId{1}, Side::Ask, Price{100}, Quantity{5}));
        extend(engine.add_limit(OrderId{2}, Side::Ask, Price{101}, Quantity{5}));
        extend(engine.add_limit(OrderId{3}, Side::Bid, Price{101}, Quantity{7}));
        engine.cancel(OrderId{2});  // (no-op now — already consumed by trade above)
        extend(engine.add_market(OrderId{4}, Side::Ask, Quantity{2}));
        return all;
    };
    plain_fills = run(plain);

    // Run identical scenario WITH scorer.
    Scorer scorer(repo_root() / "model" / "artifacts" / "model.onnx");
    auto lut = MicropriceLut::load(repo_root() / "model" / "artifacts" /
                                    "microprice_g.json");
    Engine scored(nullptr, &scorer, [](OrderId, double) {}, "AAPL", &lut);
    auto scored_fills = run(scored);

    // Fills must be identical — scorer is observer-only.
    REQUIRE(plain_fills.size() == scored_fills.size());
    for (std::size_t i = 0; i < plain_fills.size(); ++i) {
        REQUIRE(plain_fills[i].taker_id.value() == scored_fills[i].taker_id.value());
        REQUIRE(plain_fills[i].maker_id.value() == scored_fills[i].maker_id.value());
        REQUIRE(plain_fills[i].price.value() == scored_fills[i].price.value());
        REQUIRE(plain_fills[i].quantity.value() == scored_fills[i].quantity.value());
    }
}
