#pragma once

#include <functional>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "core/events/order_event.hpp"
#include "core/matching/fill.hpp"
#include "core/orderbook/book.hpp"
#include "core/scoring/feature_state.hpp"

namespace clob {

class Scorer;  // forward declare; defined in core/scoring/scorer.hpp

class Engine {
public:
    // Optional sink invoked before each accepted mutating operation. W6 wires
    // this to JournalWriter so journals replay bit-identically. Default is
    // empty -> no journaling (existing tests rely on this).
    using JournalSink = std::function<void(const OrderEvent&)>;

    // W10: optional sink invoked AFTER scoring each accepted op. Mirrors
    // JournalSink design. Score is observer-only — fills/book unchanged.
    using ScoreSink = std::function<void(OrderId, double)>;

    Engine() = default;
    explicit Engine(JournalSink sink) noexcept : journal_sink_(std::move(sink)) {}

    // W10 ctor: enables scoring path. `scorer` and `lut` are non-owning —
    // caller must keep them alive for Engine's lifetime. `ticker` is one of
    // ALL_TICKERS (used for one-hot in FeatureState).
    Engine(JournalSink journal_sink,
           Scorer* scorer,
           ScoreSink score_sink,
           std::string_view ticker,
           const MicropriceLut* lut);

    std::vector<Fill> add_limit(OrderId id, Side side, Price price, Quantity qty);
    std::vector<Fill> add_market(OrderId id, Side side, Quantity qty);
    std::vector<Fill> add_ioc(OrderId id, Side side, Price price, Quantity qty);

    bool cancel(OrderId id);
    std::vector<Fill> cancel_replace(OrderId old_id,
                                     OrderId new_id,
                                     Price price,
                                     Quantity qty);

    [[nodiscard]] const Book& book() const noexcept { return book_; }

private:
    // Inner helper used both by public add_limit and by cancel_replace. The
    // public path fires the sink before calling this; cancel_replace fires only
    // its own Replace event so the journal stays normalized (one record per
    // top-level operation, no nested events).
    std::vector<Fill> do_add_limit(OrderId id, Side side, Price price, Quantity qty);

    // W10: snapshot features + invoke scorer + fire score_sink (if all wired).
    // Returns nothing — score flows through the sink. No effect on matcher.
    void maybe_score(OrderId id);

    // W10: post-event update of FeatureState (called after every accepted mutation
    // + per-fill observe_trade for trades).
    void update_feature_state(Side taker_side, const std::vector<Fill>& fills);

    Book book_;
    JournalSink journal_sink_;

    // W10 scoring-path members. All null/empty when scoring disabled.
    Scorer* scorer_ = nullptr;           // non-owning
    ScoreSink score_sink_;
    std::unique_ptr<FeatureState> feature_state_;  // initialized only in W10 ctor
};

}  // namespace clob
