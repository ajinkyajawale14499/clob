// replay <journal-in> <fill-log-out>
//
// Deterministic replay tool. Reads OrderEvents from the input journal, feeds a
// fresh Engine, and writes every emitted Fill to the output fill log. Per ADR
// 0001, two runs on the same journal MUST produce byte-identical output.

#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <variant>

#include "core/matching/engine.hpp"
#include "io/journal/fill_log.hpp"
#include "io/journal/journal_reader.hpp"

using namespace clob;

namespace {

void apply(Engine& engine, FillLogWriter& out, const OrderEvent& ev) {
    std::visit(
        [&](const auto& e) {
            using T = std::decay_t<decltype(e)>;
            std::vector<Fill> fills;
            if constexpr (std::is_same_v<T, NewLimit>) {
                fills = engine.add_limit(e.id, e.side, e.price, e.qty);
            } else if constexpr (std::is_same_v<T, NewMarket>) {
                fills = engine.add_market(e.id, e.side, e.qty);
            } else if constexpr (std::is_same_v<T, NewIoc>) {
                fills = engine.add_ioc(e.id, e.side, e.price, e.qty);
            } else if constexpr (std::is_same_v<T, Cancel>) {
                engine.cancel(e.id);
            } else if constexpr (std::is_same_v<T, Replace>) {
                fills = engine.cancel_replace(e.old_id, e.new_id, e.price, e.qty);
            }
            for (const auto& f : fills) out.write(f);
        },
        ev);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << std::format("usage: {} <journal-in> <fill-log-out>\n", argv[0]);
        return 64;  // EX_USAGE
    }
    const std::filesystem::path journal_path{argv[1]};
    const std::filesystem::path fill_path{argv[2]};

    // Engine is constructed with NO sink — replay must not journal back into a
    // file. Doing so would create a loop if input == output.
    Engine engine;
    FillLogWriter out(fill_path);
    JournalReader in(journal_path);

    std::size_t events_read = 0;
    while (auto ev = in.next()) {
        apply(engine, out, *ev);
        ++events_read;
    }

    if (in.end_reason() == JournalReader::EndReason::TruncatedTrailingRecord) {
        std::cerr << std::format(
            "replay: WARNING — journal trailing record truncated after {} good events\n",
            events_read);
    }
    std::cout << std::format("replay: {} events -> {}\n", events_read, fill_path.string());
    return 0;
}
