#pragma once

// FeatureState — C++ mirror of model/features.py. Built up by Engine after
// each accepted op; queried (via snapshot()) BEFORE scoring the next event.
//
// Field order in ScoredFeatures MUST match model/schema.py:FEATURE_NAMES.
// Enforced by the train/serve skew test (W10 task 6.5; rtol=1e-4).
//
// All rolling stats are O(1) per observe() — ring buffers + running sums.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

#include "core/orderbook/book.hpp"
#include "core/types/price.hpp"
#include "core/types/quantity.hpp"
#include "core/types/side.hpp"

namespace clob {

// Top-of-book + L2-L5 sizes — captured from Engine's Book at snapshot time.
// Used by FeatureState::observe() to update rolling state.
struct TopOfBook {
    std::int64_t bid_price_l1 = 0;
    std::int64_t ask_price_l1 = 0;
    std::int64_t bid_size_l1 = 0;
    std::int64_t ask_size_l1 = 0;
    std::array<std::int64_t, 4> bid_size_l2_l5{};  // L2..L5
    std::array<std::int64_t, 4> ask_size_l2_l5{};
};

// Helper: extract TopOfBook from Engine's Book. Iterates the std::map<Price,Level>
// in price order (top-down for bids, bottom-up for asks). std::map iteration
// is well-defined and ADR-0001-compatible.
TopOfBook extract_tob(const Book& book);

// 19-feature vector — MUST match model/schema.py:FEATURE_NAMES order exactly.
// Float32 to match the model's input dtype.
struct ScoredFeatures {
    float microprice_g_dev = 0.0f;
    float imbalance_l1 = 0.0f;
    float spread_ticks = 0.0f;
    float spread_zscore_200 = 0.0f;
    float ofi_50 = 0.0f;
    float ofi_200 = 0.0f;
    float mlofi_l2_l5_w50 = 0.0f;
    float signed_trade_flow_50 = 0.0f;
    float tfi_50 = 0.0f;
    float realized_vol_200 = 0.0f;
    float queue_depletion_bid = 0.0f;
    float queue_depletion_ask = 0.0f;
    float ticker_AAPL = 0.0f;
    float ticker_AMZN = 0.0f;
    float ticker_GOOG = 0.0f;
    float ticker_INTC = 0.0f;
    float ticker_MSFT = 0.0f;
    float is_warm_50 = 0.0f;
    float is_warm_200 = 0.0f;

    void to_array(std::array<float, 19>& out) const noexcept;
};
static_assert(sizeof(ScoredFeatures) == 19 * sizeof(float));

// Stoikov G(I,S) lookup table — loaded from model/artifacts/microprice_g.json
// (written by model.microprice_g.MicropriceLut.save()).
class MicropriceLut {
public:
    MicropriceLut() = default;
    static MicropriceLut load(const std::filesystem::path& json_path);

    // O(1) lookup keyed on (discretized imbalance, spread in ticks).
    // imbalance ∈ [-1, +1] → bucket ∈ [0, n_imb-1]; spread_ticks → bucket.
    [[nodiscard]] double lookup(double imbalance, std::int64_t spread_ticks) const noexcept;

    [[nodiscard]] std::size_t n_imbalance_buckets() const noexcept { return n_imb_; }
    [[nodiscard]] std::size_t n_spread_buckets() const noexcept { return n_sp_; }
    [[nodiscard]] std::int64_t tick_size() const noexcept { return tick_size_; }

private:
    std::vector<double> table_;  // row-major (n_imb, n_sp)
    std::size_t n_imb_ = 0;
    std::size_t n_sp_ = 0;
    std::int64_t tick_size_ = 0;
};

// Stateful per-Engine feature accumulator. Constructor takes ticker (for one-hot
// + EWMA alpha for queue_depletion) and a reference to a pre-loaded MicropriceLut.
class FeatureState {
public:
    static constexpr std::size_t W50 = 50;
    static constexpr std::size_t W200 = 200;
    static constexpr std::size_t T50 = 50;

    FeatureState(std::string_view ticker, const MicropriceLut& lut,
                 double ewma_alpha = 0.05);

    // Snapshot the 19-feature vector at the CURRENT internal state.
    // Called by Engine BEFORE matching an event.
    [[nodiscard]] ScoredFeatures snapshot() const noexcept;

    // Update internal state with the post-event TOB. Called by Engine AFTER
    // matching an event.
    void observe(const TopOfBook& tob) noexcept;

    // Trade events feed the trade-flow features. Called by Engine for each
    // Fill produced (aggressor_side = +1 if taker is Bid, -1 if Ask).
    void observe_trade(std::int8_t aggressor_side, std::int64_t size) noexcept;

private:
    const MicropriceLut* lut_;
    double ewma_alpha_;
    std::array<float, 5> ticker_onehot_{};

    // Current snapshot — copied from last observe().
    std::int64_t bid_px_ = 0, ask_px_ = 0;
    std::int64_t bid_sz_ = 0, ask_sz_ = 0;
    std::array<std::int64_t, 4> bid_l2_l5_{}, ask_l2_l5_{};
    // First observe() seeds state but emits delta=0 to match polars
    // .diff().fill_null(0) — otherwise the first delta would be `new - 0` (huge).
    bool first_observe_ = true;

    // Ring buffer for L1 deltas → ofi_50, ofi_200.
    std::array<std::int64_t, W200> bid_delta_buf_{};
    std::array<std::int64_t, W200> ask_delta_buf_{};
    std::size_t obs_count_ = 0;             // total observe() calls

    // O(1) running sums for ofi over the last 50/200 entries.
    std::int64_t ofi_50_sum_ = 0;
    std::int64_t ofi_200_sum_ = 0;

    // Multi-level OFI (L2-L5 summed deltas), 50-event window.
    std::array<std::int64_t, W50> mlofi_buf_{};
    std::int64_t mlofi_50_sum_ = 0;

    // Realized vol — ring buffer of squared mid returns.
    std::array<double, W200> mid_ret_sq_buf_{};
    double rv_200_sum_ = 0.0;

    // Spread rolling stats (200-event window) — for spread_zscore.
    std::array<std::int64_t, W200> spread_buf_{};
    std::int64_t spread_sum_ = 0;
    std::int64_t spread_sumsq_ = 0;

    // EWMA queue depletion (single double per side).
    double queue_dep_bid_ewma_ = 0.0;
    double queue_dep_ask_ewma_ = 0.0;

    // Trade flow — separate ring buffer over last 50 trades.
    std::array<std::int64_t, T50> trade_signed_buf_{};
    std::array<std::int64_t, T50> trade_buy_buf_{};
    std::array<std::int64_t, T50> trade_sell_buf_{};
    std::size_t trade_count_ = 0;
    std::int64_t signed_trade_flow_50_sum_ = 0;
    std::int64_t trade_buy_50_sum_ = 0;
    std::int64_t trade_sell_50_sum_ = 0;
};

}  // namespace clob
