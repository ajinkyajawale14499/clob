#include "core/scoring/feature_state.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace clob;

namespace {

TopOfBook tob_at(std::int64_t bid_px, std::int64_t ask_px,
                 std::int64_t bid_sz, std::int64_t ask_sz) {
    TopOfBook t;
    t.bid_price_l1 = bid_px;
    t.ask_price_l1 = ask_px;
    t.bid_size_l1 = bid_sz;
    t.ask_size_l1 = ask_sz;
    return t;
}

}  // namespace

TEST_CASE("FeatureState: empty after construction (cold start)", "[feature_state]") {
    MicropriceLut lut;  // default-constructed; lookup() returns 0 from empty table
    FeatureState fs("AAPL", lut);
    auto f = fs.snapshot();
    REQUIRE(f.is_warm_50 == 0.0f);
    REQUIRE(f.is_warm_200 == 0.0f);
    REQUIRE(f.imbalance_l1 == 0.0f);  // bid_sz + ask_sz == 0
    REQUIRE(f.spread_ticks == 0.0f);
}

TEST_CASE("FeatureState: ticker one-hot set by ctor", "[feature_state]") {
    MicropriceLut lut;
    FeatureState fs("MSFT", lut);
    auto f = fs.snapshot();
    REQUIRE(f.ticker_MSFT == 1.0f);
    REQUIRE(f.ticker_AAPL == 0.0f);
    REQUIRE(f.ticker_AMZN == 0.0f);
    REQUIRE(f.ticker_GOOG == 0.0f);
    REQUIRE(f.ticker_INTC == 0.0f);
}

TEST_CASE("FeatureState: imbalance after observe matches expected", "[feature_state]") {
    MicropriceLut lut;
    FeatureState fs("AAPL", lut);
    // bid_sz=80, ask_sz=20 -> imbalance = (80-20)/100 = 0.6
    fs.observe(tob_at(/*bid_px=*/100, /*ask_px=*/102, /*bid_sz=*/80, /*ask_sz=*/20));
    auto f = fs.snapshot();
    REQUIRE(f.imbalance_l1 == 0.6f);
}

TEST_CASE("FeatureState: warm flags fire at expected thresholds", "[feature_state]") {
    MicropriceLut lut;
    FeatureState fs("AAPL", lut);
    auto tob = tob_at(100, 102, 50, 50);
    for (int i = 0; i < 49; ++i) fs.observe(tob);
    REQUIRE(fs.snapshot().is_warm_50 == 0.0f);
    fs.observe(tob);  // 50th
    REQUIRE(fs.snapshot().is_warm_50 == 1.0f);
    REQUIRE(fs.snapshot().is_warm_200 == 0.0f);
    for (int i = 0; i < 150; ++i) fs.observe(tob);
    REQUIRE(fs.snapshot().is_warm_200 == 1.0f);
}

TEST_CASE("FeatureState: ofi_50 tracks last-50 deltas", "[feature_state]") {
    MicropriceLut lut;
    FeatureState fs("AAPL", lut);
    // Steady state at sz=100/100.
    fs.observe(tob_at(100, 102, 100, 100));
    // Grow bid by 10 each step; ask unchanged -> Δbid=+10, Δask=0, ofi += +10
    for (int i = 1; i <= 5; ++i) {
        fs.observe(tob_at(100, 102, 100 + i * 10, 100));
    }
    auto f = fs.snapshot();
    // After first observe Δ=0 (was init from 0); then 5 deltas of +10 each = 50
    REQUIRE(f.ofi_50 == 50.0f);
}

TEST_CASE("FeatureState: realized_vol_200 = sqrt of summed squared returns",
          "[feature_state]") {
    MicropriceLut lut;
    FeatureState fs("AAPL", lut);
    // First observe: ret=0 (no prev mid per Python diff.fill_null(0)).
    // Then 3 returns of +2, -2, +2 — squared sum = 12; sqrt(12) ≈ 3.464.
    fs.observe(tob_at(99, 101, 50, 50));    // mid=100, first -> ret_sq=0
    fs.observe(tob_at(101, 103, 50, 50));   // mid=102, ret=+2
    fs.observe(tob_at(99, 101, 50, 50));    // mid=100, ret=-2
    fs.observe(tob_at(101, 103, 50, 50));   // mid=102, ret=+2
    auto f = fs.snapshot();
    REQUIRE(std::abs(f.realized_vol_200 - std::sqrt(12.0f)) < 1e-3f);
}

TEST_CASE("FeatureState: queue_depletion EWMA tracks direction", "[feature_state]") {
    MicropriceLut lut;
    FeatureState fs("AAPL", lut);
    fs.observe(tob_at(100, 102, 100, 100));
    // Grow bid sequentially -> positive EWMA on bid side
    for (int i = 1; i <= 10; ++i) {
        fs.observe(tob_at(100, 102, 100 + i * 10, 100));
    }
    auto f = fs.snapshot();
    REQUIRE(f.queue_depletion_bid > 0.0f);
    REQUIRE(std::abs(f.queue_depletion_ask) < 0.1f);  // ask side unchanged
}

TEST_CASE("FeatureState: trade-flow features track trades", "[feature_state]") {
    MicropriceLut lut;
    FeatureState fs("AAPL", lut);
    fs.observe_trade(1, 100);   // buy aggressor
    fs.observe_trade(1, 50);    // buy aggressor
    fs.observe_trade(-1, 30);   // sell aggressor
    auto f = fs.snapshot();
    // signed_trade_flow_50 = 100 + 50 - 30 = 120
    REQUIRE(f.signed_trade_flow_50 == 120.0f);
    // tfi_50 = (150 - 30) / 180 ≈ 0.6667
    REQUIRE(std::abs(f.tfi_50 - (120.0f / 180.0f)) < 1e-4f);
}

TEST_CASE("FeatureState: spread_ticks reflects current spread", "[feature_state]") {
    MicropriceLut lut;
    FeatureState fs("AAPL", lut);
    // spread = ask - bid = 102 - 100 = 2; lut tick_size = 0 (default) -> divide by zero guard
    fs.observe(tob_at(100, 102, 50, 50));
    auto f = fs.snapshot();
    // With default lut.tick_size_=0, spread_ticks stays at 0 (guarded).
    REQUIRE(f.spread_ticks == 0.0f);
}
