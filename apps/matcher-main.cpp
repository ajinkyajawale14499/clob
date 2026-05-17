// matcher-cli — interactive matching-engine REPL with optional ML scoring.
//
// Usage:
//   matcher-cli
//   matcher-cli --journal=PATH                      # journal every accepted op
//   matcher-cli --model=ONNX --lut=JSON [--ticker=AAPL]
//                                                    # enable hot-path scoring;
//                                                    # prints score for each op

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
#include "core/scoring/feature_state.hpp"
#include "core/scoring/scorer.hpp"
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

struct Args {
    std::string journal_path;
    std::string model_path;
    std::string lut_path;
    std::string ticker = "AAPL";
};

bool parse_flag(std::string_view a, std::string_view key, std::string& out) {
    if (a.starts_with(key)) {
        out.assign(a.substr(key.size()));
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a{argv[i]};
        if (parse_flag(a, "--journal=", args.journal_path)) continue;
        if (parse_flag(a, "--model=",   args.model_path))   continue;
        if (parse_flag(a, "--lut=",     args.lut_path))     continue;
        if (parse_flag(a, "--ticker=",  args.ticker))       continue;
        std::cerr << std::format("unknown arg: {}\n", a);
        std::cerr << "usage: matcher-cli [--journal=PATH] "
                     "[--model=ONNX --lut=JSON] [--ticker=TICKER]\n";
        return 64;
    }

    if (!args.model_path.empty() != !args.lut_path.empty()) {
        std::cerr << "--model and --lut must be specified together\n";
        return 64;
    }

    // Optional journal sink.
    std::unique_ptr<JournalWriter> journal;
    Engine::JournalSink journal_sink;
    if (!args.journal_path.empty()) {
        journal = std::make_unique<JournalWriter>(args.journal_path);
        journal_sink = [&](const OrderEvent& ev) { journal->write(ev); };
    }

    // Optional scorer + LUT + score sink.
    std::unique_ptr<Scorer> scorer;
    std::unique_ptr<MicropriceLut> lut;
    Engine::ScoreSink score_sink;
    if (!args.model_path.empty()) {
        scorer = std::make_unique<Scorer>(args.model_path);
        lut = std::make_unique<MicropriceLut>(MicropriceLut::load(args.lut_path));
        score_sink = [](OrderId id, double s) {
            std::cout << std::format("  score: id={} P(Up)-P(Down)={:+.4f}\n",
                                     id.value(), s);
        };
    }

    // Banner.
    std::cout << "clob CLI";
    if (journal) std::cout << " journal=" << args.journal_path;
    if (scorer)  std::cout << " model=" << args.model_path << " ticker=" << args.ticker;
    std::cout << "\n  commands: limit | market | ioc | cancel | book | quit\n";

    // Build the engine — pick the right ctor depending on which sinks are wired.
    std::unique_ptr<Engine> engine;
    if (scorer) {
        engine = std::make_unique<Engine>(
            std::move(journal_sink), scorer.get(),
            std::move(score_sink), args.ticker, lut.get());
    } else if (journal) {
        engine = std::make_unique<Engine>(std::move(journal_sink));
    } else {
        engine = std::make_unique<Engine>();
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "quit" || cmd == "exit") break;

        if (cmd == "book") {
            print_book(engine->book());
        } else if (cmd == "limit") {
            std::uint64_t id;
            std::string s;
            std::int64_t price, qty;
            iss >> id >> s >> price >> qty;
            print_fills(engine->add_limit(OrderId{id}, parse_side(s),
                                          Price{price}, Quantity{qty}));
            print_book(engine->book());
        } else if (cmd == "market") {
            std::uint64_t id;
            std::string s;
            std::int64_t qty;
            iss >> id >> s >> qty;
            print_fills(engine->add_market(OrderId{id}, parse_side(s), Quantity{qty}));
            print_book(engine->book());
        } else if (cmd == "ioc") {
            std::uint64_t id;
            std::string s;
            std::int64_t price, qty;
            iss >> id >> s >> price >> qty;
            print_fills(engine->add_ioc(OrderId{id}, parse_side(s),
                                        Price{price}, Quantity{qty}));
            print_book(engine->book());
        } else if (cmd == "cancel") {
            std::uint64_t id;
            iss >> id;
            std::cout << std::format("  {}: {}\n",
                                     engine->cancel(OrderId{id}) ? "cancelled" : "not found",
                                     id);
        } else if (!cmd.empty()) {
            std::cout << std::format("  unknown: {}\n", cmd);
        }
    }
    return 0;
}
