// matcher-cli — interactive matching-engine REPL.
//
// Usage:
//   matcher-cli                       # no journaling
//   matcher-cli --journal=PATH        # append every accepted op to PATH

#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/matching/engine.hpp"
#include "io/journal/journal_writer.hpp"

using namespace clob;

namespace {

Side parse_side(std::string_view s) {
    return (s == "bid" || s == "buy") ? Side::Bid : Side::Ask;
}

void print_book(const Book& book) {
    auto bb = book.best_bid();
    auto ba = book.best_ask();
    std::cout << std::format("Book: bid={}  ask={}\n",
                             bb ? std::to_string(bb->value()) : "-",
                             ba ? std::to_string(ba->value()) : "-");
}

void print_fills(const std::vector<Fill>& fills) {
    for (const auto& f : fills) {
        std::cout << std::format("  fill: maker={} taker={} price={} qty={}\n",
                                 f.maker_id.value(), f.taker_id.value(),
                                 f.price.value(), f.quantity.value());
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string journal_path;
    for (int i = 1; i < argc; ++i) {
        constexpr std::string_view kFlag = "--journal=";
        std::string_view a{argv[i]};
        if (a.starts_with(kFlag)) {
            journal_path.assign(a.substr(kFlag.size()));
        } else {
            std::cerr << std::format("unknown arg: {}\n", a);
            return 64;
        }
    }

    std::unique_ptr<JournalWriter> journal;
    Engine::JournalSink sink;
    if (!journal_path.empty()) {
        journal = std::make_unique<JournalWriter>(journal_path);
        sink = [&](const OrderEvent& ev) { journal->write(ev); };
        std::cout << std::format("clob CLI (journaling to {}) — commands: "
                                 "limit | market | ioc | cancel | book | quit\n",
                                 journal_path);
    } else {
        std::cout << "clob CLI — commands: limit | market | ioc | cancel | book | quit\n";
    }
    Engine engine(sink);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "quit" || cmd == "exit") break;

        if (cmd == "book") {
            print_book(engine.book());
        } else if (cmd == "limit") {
            std::uint64_t id;
            std::string s;
            std::int64_t price, qty;
            iss >> id >> s >> price >> qty;
            print_fills(engine.add_limit(OrderId{id}, parse_side(s),
                                         Price{price}, Quantity{qty}));
            print_book(engine.book());
        } else if (cmd == "market") {
            std::uint64_t id;
            std::string s;
            std::int64_t qty;
            iss >> id >> s >> qty;
            print_fills(engine.add_market(OrderId{id}, parse_side(s), Quantity{qty}));
            print_book(engine.book());
        } else if (cmd == "ioc") {
            std::uint64_t id;
            std::string s;
            std::int64_t price, qty;
            iss >> id >> s >> price >> qty;
            print_fills(engine.add_ioc(OrderId{id}, parse_side(s),
                                       Price{price}, Quantity{qty}));
            print_book(engine.book());
        } else if (cmd == "cancel") {
            std::uint64_t id;
            iss >> id;
            std::cout << std::format("  {}: {}\n",
                                     engine.cancel(OrderId{id}) ? "cancelled" : "not found",
                                     id);
        } else if (!cmd.empty()) {
            std::cout << std::format("  unknown: {}\n", cmd);
        }
    }
    return 0;
}
