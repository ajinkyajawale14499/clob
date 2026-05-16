#include <cstdint>
#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/matching/engine.hpp"

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

int main() {
    Engine engine;
    std::cout << "clob CLI — commands: limit | market | ioc | cancel | book | quit\n";

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
