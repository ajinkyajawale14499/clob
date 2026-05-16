#include "core/scoring/feature_state.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace clob {

namespace {

// L1 + L2-L5 size extraction helpers — iterate the std::map deterministically.
// Returns 0 in slots where the level doesn't exist (book is shallower than 5).
template <typename MapT>
void fill_top_n(const MapT& m, std::int64_t& px_l1, std::int64_t& sz_l1,
                std::array<std::int64_t, 4>& l2_l5) {
    px_l1 = 0;
    sz_l1 = 0;
    l2_l5.fill(0);
    auto it = m.begin();
    if (it == m.end()) return;
    px_l1 = it->first.value();
    sz_l1 = it->second.total_quantity().value();
    ++it;
    for (std::size_t i = 0; i < 4 && it != m.end(); ++i, ++it) {
        l2_l5[i] = it->second.total_quantity().value();
    }
}

}  // namespace

TopOfBook extract_tob(const Book& book) {
    TopOfBook tob;
    // The Book exposes best_bid/best_ask but not full iteration; we need
    // to add a helper or use a friend trick. For now this requires friending
    // — see book.hpp. Simpler approach: add public iteration helpers.
    // For W10 we'll use a minimal accessor pattern.
    // (Book::for_each_bid / for_each_ask not yet implemented — TODO if perf
    //  matters; for now we just expose L1 via best_bid/best_ask and use
    //  Book::level_at for L2-L5 if needed. v1 sets L2-L5 sizes to 0 if not
    //  reachable via the public API.)
    auto bb = book.best_bid();
    auto ba = book.best_ask();
    if (bb) {
        tob.bid_price_l1 = bb->value();
        const Level* lvl = book.level_at(Side::Bid, *bb);
        if (lvl) tob.bid_size_l1 = lvl->total_quantity().value();
    }
    if (ba) {
        tob.ask_price_l1 = ba->value();
        const Level* lvl = book.level_at(Side::Ask, *ba);
        if (lvl) tob.ask_size_l1 = lvl->total_quantity().value();
    }
    // L2-L5: leave at zero in this simple accessor (v1 limitation; train/serve
    // skew test acknowledges that mlofi_l2_l5_w50 may diverge between Python
    // (real LOBSTER L2-L5) and C++ (Engine state only has matched orders).
    // For backtest this is fine — LOBSTER replay populates the Engine's Book
    // with L2+ levels too, just not exposed via Book yet. v2 work.
    return tob;
}

// --- ScoredFeatures::to_array — packs in canonical order (matches schema.py).
void ScoredFeatures::to_array(std::array<float, 19>& out) const noexcept {
    // Use the struct layout directly; static_assert above guarantees no padding.
    std::memcpy(out.data(), this, sizeof(ScoredFeatures));
}

// --- MicropriceLut — loads the JSON written by model.microprice_g.

MicropriceLut MicropriceLut::load(const std::filesystem::path& json_path) {
    std::ifstream f(json_path);
    if (!f) {
        throw std::runtime_error("MicropriceLut: cannot open " + json_path.string());
    }
    nlohmann::json j;
    f >> j;

    MicropriceLut lut;
    lut.n_imb_ = j.at("n_imbalance_buckets").get<std::size_t>();
    lut.n_sp_ = j.at("n_spread_buckets").get<std::size_t>();
    lut.tick_size_ = j.at("tick_size").get<std::int64_t>();

    const auto& table = j.at("table");
    if (!table.is_array() || table.size() != lut.n_imb_) {
        throw std::runtime_error("MicropriceLut: invalid table shape");
    }
    lut.table_.reserve(lut.n_imb_ * lut.n_sp_);
    for (std::size_t i = 0; i < lut.n_imb_; ++i) {
        const auto& row = table.at(i);
        if (!row.is_array() || row.size() != lut.n_sp_) {
            throw std::runtime_error("MicropriceLut: invalid table row");
        }
        for (std::size_t s = 0; s < lut.n_sp_; ++s) {
            lut.table_.push_back(row.at(s).get<double>());
        }
    }
    return lut;
}

double MicropriceLut::lookup(double imbalance, std::int64_t spread_ticks) const noexcept {
    if (table_.empty()) return 0.0;
    // imbalance ∈ [-1, +1] → bucket ∈ [0, n_imb-1]. Match Python's int((imb+1)/2*N).
    std::int64_t i = static_cast<std::int64_t>((imbalance + 1.0) / 2.0 * static_cast<double>(n_imb_));
    if (i < 0) i = 0;
    if (i >= static_cast<std::int64_t>(n_imb_)) i = static_cast<std::int64_t>(n_imb_) - 1;
    // spread_ticks: 1 → bucket 0, 2 → bucket 1, ...
    std::int64_t s = spread_ticks - 1;
    if (s < 0) s = 0;
    if (s >= static_cast<std::int64_t>(n_sp_)) s = static_cast<std::int64_t>(n_sp_) - 1;
    return table_[static_cast<std::size_t>(i) * n_sp_ + static_cast<std::size_t>(s)];
}

// --- FeatureState

FeatureState::FeatureState(std::string_view ticker, const MicropriceLut& lut,
                            double ewma_alpha)
    : lut_(&lut), ewma_alpha_(ewma_alpha) {
    // Set ticker one-hot. Order matches model/schema.py: AAPL, AMZN, GOOG, INTC, MSFT.
    constexpr std::array<std::string_view, 5> tickers = {
        "AAPL", "AMZN", "GOOG", "INTC", "MSFT",
    };
    for (std::size_t i = 0; i < tickers.size(); ++i) {
        ticker_onehot_[i] = (ticker == tickers[i]) ? 1.0f : 0.0f;
    }
    bid_delta_buf_.fill(0);
    ask_delta_buf_.fill(0);
    mlofi_buf_.fill(0);
    mid_ret_sq_buf_.fill(0.0);
    spread_buf_.fill(0);
    trade_signed_buf_.fill(0);
    trade_buy_buf_.fill(0);
    trade_sell_buf_.fill(0);
}

ScoredFeatures FeatureState::snapshot() const noexcept {
    ScoredFeatures f;
    // Book-shape features.
    const std::int64_t total = bid_sz_ + ask_sz_;
    if (total > 0) {
        f.imbalance_l1 = static_cast<float>(bid_sz_ - ask_sz_) / static_cast<float>(total);
    } else {
        f.imbalance_l1 = 0.0f;
    }
    const std::int64_t spread_native = ask_px_ - bid_px_;
    if (lut_->tick_size() > 0) {
        f.spread_ticks = static_cast<float>(spread_native) / static_cast<float>(lut_->tick_size());
    }

    // spread_zscore_200 — only meaningful when warm (need W200 spreads).
    if (obs_count_ >= W200) {
        const double mean = static_cast<double>(spread_sum_) / static_cast<double>(W200);
        const double var = static_cast<double>(spread_sumsq_) / static_cast<double>(W200) - mean * mean;
        if (var > 1e-12) {
            f.spread_zscore_200 =
                static_cast<float>((static_cast<double>(spread_native) - mean) / std::sqrt(var));
        }
    }

    // microprice_g_dev — LUT lookup using current imbalance + spread_ticks (rounded down).
    {
        const double imb = static_cast<double>(f.imbalance_l1);
        const std::int64_t sp = static_cast<std::int64_t>(f.spread_ticks);
        f.microprice_g_dev = static_cast<float>(lut_->lookup(imb, sp));
    }

    // ofi_50 / ofi_200 / mlofi_l2_l5_w50.
    f.ofi_50 = static_cast<float>(ofi_50_sum_);
    f.ofi_200 = static_cast<float>(ofi_200_sum_);
    f.mlofi_l2_l5_w50 = static_cast<float>(mlofi_50_sum_);

    // Trade-flow features.
    f.signed_trade_flow_50 = static_cast<float>(signed_trade_flow_50_sum_);
    const std::int64_t total_trade_vol = trade_buy_50_sum_ + trade_sell_50_sum_;
    if (total_trade_vol > 0) {
        f.tfi_50 = static_cast<float>(trade_buy_50_sum_ - trade_sell_50_sum_) /
                    static_cast<float>(total_trade_vol);
    }

    // realized_vol_200 — sqrt of ring buffer sum of squared mid returns.
    if (rv_200_sum_ > 0.0) {
        f.realized_vol_200 = static_cast<float>(std::sqrt(rv_200_sum_));
    }

    // EWMA queue depletion.
    f.queue_depletion_bid = static_cast<float>(queue_dep_bid_ewma_);
    f.queue_depletion_ask = static_cast<float>(queue_dep_ask_ewma_);

    // Ticker one-hot.
    f.ticker_AAPL = ticker_onehot_[0];
    f.ticker_AMZN = ticker_onehot_[1];
    f.ticker_GOOG = ticker_onehot_[2];
    f.ticker_INTC = ticker_onehot_[3];
    f.ticker_MSFT = ticker_onehot_[4];

    // Warm flags.
    f.is_warm_50 = (obs_count_ >= W50) ? 1.0f : 0.0f;
    f.is_warm_200 = (obs_count_ >= W200) ? 1.0f : 0.0f;

    return f;
}

void FeatureState::observe(const TopOfBook& tob) noexcept {
    // Compute deltas vs previous state.
    // First observe seeds state; deltas = 0 (mirrors polars .diff().fill_null(0)).
    const bool first = first_observe_;
    const std::int64_t d_bid_l1 = first ? 0 : (tob.bid_size_l1 - bid_sz_);
    const std::int64_t d_ask_l1 = first ? 0 : (tob.ask_size_l1 - ask_sz_);

    // L1 OFI = Δbid_size - Δask_size.
    const std::int64_t l1_ofi_delta = d_bid_l1 - d_ask_l1;

    // Multi-level L2-L5 OFI: sum of (Δbid_lv - Δask_lv).
    std::int64_t l25_ofi_delta = 0;
    if (!first) {
        for (std::size_t i = 0; i < 4; ++i) {
            l25_ofi_delta += (tob.bid_size_l2_l5[i] - bid_l2_l5_[i]) -
                              (tob.ask_size_l2_l5[i] - ask_l2_l5_[i]);
        }
    }

    // Update ring buffers + running sums.
    const std::size_t pos50 = obs_count_ % W50;
    const std::size_t pos200 = obs_count_ % W200;

    // ofi_200 (using bid_delta_buf_/ask_delta_buf_ — sized to W200).
    ofi_200_sum_ += l1_ofi_delta - bid_delta_buf_[pos200] + ask_delta_buf_[pos200];
    // ofi_50 — same buffer, but track separately since the window is different.
    // We use a trick: for the 50-window, the oldest slot is `obs_count - 50`.
    if (obs_count_ >= W50) {
        const std::size_t evict = (obs_count_ - W50) % W200;
        ofi_50_sum_ -= bid_delta_buf_[evict] - ask_delta_buf_[evict];
    }
    ofi_50_sum_ += l1_ofi_delta;
    // Now overwrite the W200 buffer slot for next time.
    bid_delta_buf_[pos200] = d_bid_l1;
    ask_delta_buf_[pos200] = d_ask_l1;

    // mlofi_l2_l5_w50 — sums into mlofi_buf_ (sized W50).
    mlofi_50_sum_ += l25_ofi_delta - mlofi_buf_[pos50];
    mlofi_buf_[pos50] = l25_ofi_delta;

    // realized_vol_200: ring buffer of squared mid-returns. First observe
    // has no previous mid, so ret_sq=0 (mirrors polars .diff().fill_null(0)).
    const double new_mid = static_cast<double>(tob.bid_price_l1 + tob.ask_price_l1) / 2.0;
    double mid_ret_sq = 0.0;
    if (!first) {
        const double prev_mid = static_cast<double>(bid_px_ + ask_px_) / 2.0;
        mid_ret_sq = (new_mid - prev_mid) * (new_mid - prev_mid);
    }
    rv_200_sum_ += mid_ret_sq - mid_ret_sq_buf_[pos200];
    mid_ret_sq_buf_[pos200] = mid_ret_sq;

    // Spread rolling sum + sum-of-squares for z-score.
    const std::int64_t spread = tob.ask_price_l1 - tob.bid_price_l1;
    spread_sum_ += spread - spread_buf_[pos200];
    spread_sumsq_ += spread * spread - spread_buf_[pos200] * spread_buf_[pos200];
    spread_buf_[pos200] = spread;

    // EWMA queue depletion.
    queue_dep_bid_ewma_ = ewma_alpha_ * static_cast<double>(d_bid_l1)
                          + (1.0 - ewma_alpha_) * queue_dep_bid_ewma_;
    queue_dep_ask_ewma_ = ewma_alpha_ * static_cast<double>(d_ask_l1)
                          + (1.0 - ewma_alpha_) * queue_dep_ask_ewma_;

    // Persist current snapshot.
    bid_px_ = tob.bid_price_l1;
    ask_px_ = tob.ask_price_l1;
    bid_sz_ = tob.bid_size_l1;
    ask_sz_ = tob.ask_size_l1;
    bid_l2_l5_ = tob.bid_size_l2_l5;
    ask_l2_l5_ = tob.ask_size_l2_l5;
    first_observe_ = false;
    ++obs_count_;
}

void FeatureState::observe_trade(std::int8_t aggressor_side, std::int64_t size) noexcept {
    const std::int64_t signed_size = static_cast<std::int64_t>(aggressor_side) * size;
    const std::int64_t buy_size = (aggressor_side == 1) ? size : 0;
    const std::int64_t sell_size = (aggressor_side == -1) ? size : 0;

    const std::size_t pos = trade_count_ % T50;
    signed_trade_flow_50_sum_ += signed_size - trade_signed_buf_[pos];
    trade_buy_50_sum_ += buy_size - trade_buy_buf_[pos];
    trade_sell_50_sum_ += sell_size - trade_sell_buf_[pos];

    trade_signed_buf_[pos] = signed_size;
    trade_buy_buf_[pos] = buy_size;
    trade_sell_buf_[pos] = sell_size;
    ++trade_count_;
}

}  // namespace clob
