#include "core/matching/engine.hpp"

#include <algorithm>  // std::min
#include <optional>
#include <utility>

#include "core/scoring/scorer.hpp"

namespace clob {

// W10 ctor: enables scoring path.
Engine::Engine(JournalSink journal_sink,
                Scorer* scorer,
                ScoreSink score_sink,
                std::string_view ticker,
                const MicropriceLut* lut)
    : journal_sink_(std::move(journal_sink)),
      scorer_(scorer),
      score_sink_(std::move(score_sink)),
      feature_state_(lut ? std::make_unique<FeatureState>(ticker, *lut) : nullptr) {}

void Engine::maybe_score(OrderId id) {
    if (!scorer_ || !score_sink_ || !feature_state_) return;
    const ScoredFeatures f = feature_state_->snapshot();
    const double s = scorer_->score(f);
    score_sink_(id, s);
}

void Engine::update_feature_state(Side taker_side,
                                   const std::vector<Fill>& fills) {
    if (!feature_state_) return;
    // After-event TOB snapshot.
    feature_state_->observe(extract_tob(book_));
    // Per-fill trade observation. The aggressor is the taker (the order that
    // crossed); maker is the resting order.
    const std::int8_t aggressor = (taker_side == Side::Bid) ? 1 : -1;
    for (const auto& f : fills) {
        feature_state_->observe_trade(aggressor, f.quantity.value());
    }
}

namespace {

// Match `qty` of taker against opposite side, optionally capped at price_cap.
// price_cap = nullopt -> market order (walk to empty book).
std::vector<Fill> match_against(Book& book,
                                OrderId taker,
                                Side taker_side,
                                std::optional<Price> price_cap,
                                Quantity qty) {
    std::vector<Fill> fills;
    Quantity remaining = qty;
    const Side opp = opposite(taker_side);

    auto top_opp = [&]() -> std::optional<Price> {
        return taker_side == Side::Bid ? book.best_ask() : book.best_bid();
    };

    auto in_cap = [&](Price p) -> bool {
        if (!price_cap) return true;
        return taker_side == Side::Bid ? p.value() <= price_cap->value()
                                       : p.value() >= price_cap->value();
    };

    while (remaining.value() > 0) {
        auto top = top_opp();
        if (!top || !in_cap(*top)) break;
        const Price match_price = *top;

        for (Level* lvl = book.level_at(opp, match_price);
             lvl != nullptr && remaining.value() > 0 && !lvl->empty();
             lvl = book.level_at(opp, match_price)) {
            const Order head_copy = lvl->front();
            const Quantity take{std::min(remaining.value(), head_copy.quantity.value())};
            fills.push_back({taker, head_copy.id, match_price, take});
            remaining = Quantity{remaining.value() - take.value()};

            if (auto consumed = lvl->consume_front(take); consumed) {
                book.unindex(*consumed);
            }
        }
        book.drop_if_empty(opp, match_price);
    }
    return fills;
}

}  // namespace

std::vector<Fill> Engine::do_add_limit(OrderId id, Side side, Price price, Quantity qty) {
    auto fills = match_against(book_, id, side, price, qty);
    std::int64_t consumed = 0;
    for (const auto& f : fills) consumed += f.quantity.value();
    const Quantity remaining{qty.value() - consumed};
    if (remaining.value() > 0) {
        // Reject duplicate resting OrderId (see Task 3.4 rationale).
        if (book_.find(id).has_value()) return fills;
        book_.add_limit(price, Order{id, side, remaining});
    }
    return fills;
}

std::vector<Fill> Engine::add_limit(OrderId id, Side side, Price price, Quantity qty) {
    if (qty.value() <= 0) return {};
    if (journal_sink_) journal_sink_(NewLimit{id, side, price, qty});
    maybe_score(id);
    auto fills = do_add_limit(id, side, price, qty);
    update_feature_state(side, fills);
    return fills;
}

std::vector<Fill> Engine::add_market(OrderId id, Side side, Quantity qty) {
    if (qty.value() <= 0) return {};
    if (journal_sink_) journal_sink_(NewMarket{id, side, qty});
    maybe_score(id);
    auto fills = match_against(book_, id, side, std::nullopt, qty);
    update_feature_state(side, fills);
    return fills;
}

std::vector<Fill> Engine::add_ioc(OrderId id, Side side, Price price, Quantity qty) {
    if (qty.value() <= 0) return {};
    if (journal_sink_) journal_sink_(NewIoc{id, side, price, qty});
    maybe_score(id);
    auto fills = match_against(book_, id, side, price, qty);
    update_feature_state(side, fills);
    return fills;
}

bool Engine::cancel(OrderId id) {
    // Journal only successful cancels — unknown-id is a no-op and shouldn't
    // bloat the log or change replay state.
    if (!book_.find(id).has_value()) return false;
    if (journal_sink_) journal_sink_(Cancel{id});
    maybe_score(id);
    const bool ok = book_.cancel(id);
    // Cancel doesn't produce fills, but the book state changed so we still
    // observe() the new TOB for future feature snapshots.
    if (feature_state_) feature_state_->observe(extract_tob(book_));
    return ok;
}

std::vector<Fill> Engine::cancel_replace(OrderId old_id,
                                         OrderId new_id,
                                         Price price,
                                         Quantity qty) {
    // Validate BEFORE cancel — otherwise we destroy old_id and then fail to replace,
    // leaving the client with no order at all (worse than rejecting outright).
    if (qty.value() <= 0) return {};
    auto loc = book_.find(old_id);
    if (!loc) return {};
    if (old_id != new_id && book_.find(new_id).has_value()) return {};  // duplicate target id

    if (journal_sink_) journal_sink_(Replace{old_id, new_id, price, qty});
    maybe_score(new_id);

    const Side side = loc->side;
    book_.cancel(old_id);
    // Use do_add_limit (not public add_limit) so the sink only fires once per
    // top-level op — Replace, not Replace + NewLimit.
    auto fills = do_add_limit(new_id, side, price, qty);
    update_feature_state(side, fills);
    return fills;
}

}  // namespace clob
