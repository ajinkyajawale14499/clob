#include "core/matching/engine.hpp"

#include <catch2/catch_test_macros.hpp>

// rapidcheck's Gen.hpp uses std::current_exception without including <exception>.
// Include it explicitly before rapidcheck headers.
#include <exception>

#include <rapidcheck.h>

#include <cstdint>
#include <set>
#include <vector>

using clob::Engine;
using clob::OrderId;
using clob::Price;
using clob::Quantity;
using clob::Side;

namespace {

enum class OpKind { AddLimit, AddMarket, AddIoc, Cancel };

struct Op {
    OpKind kind;
    std::uint64_t id;
    Side side;
    std::int64_t price;
    std::int64_t qty;
};

}  // namespace

namespace rc {
template <>
struct Arbitrary<Op> {
    static Gen<Op> arbitrary() {
        return gen::build<Op>(
            gen::set(&Op::kind,
                     gen::element(OpKind::AddLimit, OpKind::AddMarket,
                                  OpKind::AddIoc, OpKind::Cancel)),
            gen::set(&Op::id, gen::inRange<std::uint64_t>(1, 200)),
            gen::set(&Op::side, gen::element(Side::Bid, Side::Ask)),
            gen::set(&Op::price, gen::inRange<std::int64_t>(9000, 11000)),
            gen::set(&Op::qty, gen::inRange<std::int64_t>(1, 50)));
    }
};
}  // namespace rc

namespace {

void apply(Engine& e, std::set<std::uint64_t>& live_ids, const Op& op) {
    switch (op.kind) {
        case OpKind::AddLimit:
            if (live_ids.insert(op.id).second) {
                e.add_limit(OrderId{op.id}, op.side, Price{op.price}, Quantity{op.qty});
            }
            break;
        case OpKind::AddMarket:
            e.add_market(OrderId{op.id}, op.side, Quantity{op.qty});
            break;
        case OpKind::AddIoc:
            e.add_ioc(OrderId{op.id}, op.side, Price{op.price}, Quantity{op.qty});
            break;
        case OpKind::Cancel:
            if (e.cancel(OrderId{op.id})) live_ids.erase(op.id);
            break;
    }
}

}  // namespace

TEST_CASE("Property: bid_top < ask_top never violated", "[property]") {
    rc::check([](const std::vector<Op>& ops) {
        Engine e;
        std::set<std::uint64_t> live;
        for (const auto& op : ops) {
            apply(e, live, op);
            auto bb = e.book().best_bid();
            auto ba = e.book().best_ask();
            if (bb && ba) RC_ASSERT(bb->value() < ba->value());
        }
    });
}

TEST_CASE("Property: id_index_ consistency — find() agrees with level contents", "[property]") {
    // STRONG version (v2.1): if find(id) says (side, price), the level at (side, price)
    // MUST actually contain id. Catches stale id_index_ entries after fills consume makers.
    rc::check([](const std::vector<Op>& ops) {
        Engine e;
        std::set<std::uint64_t> live;
        for (const auto& op : ops) {
            apply(e, live, op);
        }
        for (auto id : live) {
            auto loc = e.book().find(OrderId{id});
            if (loc) {
                const auto* lvl = e.book().level_at(loc->side, loc->price);
                RC_ASSERT(lvl != nullptr);
                RC_ASSERT(lvl->contains(OrderId{id}));
            }
        }
    });
}

TEST_CASE("Property: Cancel twice returns true then false", "[property]") {
    rc::check([](std::uint64_t id_seed, std::int64_t price, std::int64_t qty) {
        if (qty <= 0 || price < 1000 || price > 20000) return;
        Engine e;
        e.add_limit(OrderId{id_seed}, Side::Bid, Price{price}, Quantity{qty});
        RC_ASSERT(e.cancel(OrderId{id_seed}));
        RC_ASSERT(!e.cancel(OrderId{id_seed}));
    });
}

TEST_CASE("Property: top-of-book levels are never empty", "[property]") {
    // (v2.1) Book doesn't expose level iteration so we only check BBO. The
    // structural invariant ("drop_if_empty is called on every removal site") is
    // verified via unit tests in test_book + test_engine, not property testing.
    rc::check([](const std::vector<Op>& ops) {
        Engine e;
        std::set<std::uint64_t> live;
        for (const auto& op : ops) {
            apply(e, live, op);
            if (auto bb = e.book().best_bid()) {
                const auto* lvl = e.book().level_at(Side::Bid, *bb);
                RC_ASSERT(lvl != nullptr && lvl->total_quantity().value() > 0);
            }
            if (auto ba = e.book().best_ask()) {
                const auto* lvl = e.book().level_at(Side::Ask, *ba);
                RC_ASSERT(lvl != nullptr && lvl->total_quantity().value() > 0);
            }
        }
    });
}
