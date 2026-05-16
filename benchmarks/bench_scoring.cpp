// bench-scoring — HdrHistogram-based latency benchmark for the scored Engine path.
//
// Measures: end-to-end scored `Engine::add_limit` wall-clock latency, which
// includes FeatureState snapshot + ONNX inference + ScoreSink callback +
// match_against + FeatureState observe (the full hot path).
//
// SLO (ADR 0001 amendment + plan v3 Q5): p99 < 1ms over 100k warmed calls
// after 10k warmup. Excludes IO/startup/model-load. Reports p50/p90/p99/p999/max
// to stdout and writes docs/bench.md (overwritten each run).
//
// Usage: ./bench-scoring [model_path] [lut_path] [output_md_path]
// Exits non-zero if p99 >= 1ms.

#include <hdr/hdr_histogram.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "core/matching/engine.hpp"
#include "core/scoring/scorer.hpp"

using namespace clob;
using hires = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
    const std::string model_path =
        argc > 1 ? argv[1] : "model/artifacts/model.onnx";
    const std::string lut_path =
        argc > 2 ? argv[2] : "model/artifacts/microprice_g.json";
    const std::string md_path =
        argc > 3 ? argv[3] : "docs/bench.md";

    if (!std::filesystem::exists(model_path) || !std::filesystem::exists(lut_path)) {
        std::cerr << "bench-scoring: missing artifacts. Run `uv run python -m model.train`."
                  << "\n  expected: " << model_path << " + " << lut_path << "\n";
        return 2;
    }

    std::cout << "Loading scorer + LUT..." << std::flush;
    Scorer scorer(model_path);
    auto lut = MicropriceLut::load(lut_path);
    std::cout << " done.\n";

    Engine engine(nullptr, &scorer, [](OrderId, double) {}, "AAPL", &lut);

    // Seed book with 1000 limit orders on both sides.
    std::cout << "Seeding book (1k orders)..." << std::flush;
    for (std::uint64_t i = 1; i <= 1000; ++i) {
        const Side s = (i % 2 == 0) ? Side::Bid : Side::Ask;
        const std::int64_t base = (s == Side::Bid) ? 9950 : 10050;
        const std::int64_t off = static_cast<std::int64_t>((i % 50));
        const std::int64_t px = base + (s == Side::Bid ? -off : off);
        engine.add_limit(OrderId{i}, s, Price{px}, Quantity{1 + static_cast<std::int64_t>(i % 20)});
    }
    std::cout << " done.\n";

    // Warmup: 10k scored ops.
    std::cout << "Warmup (10k ops)..." << std::flush;
    constexpr std::uint64_t kWarmup = 10'000;
    constexpr std::uint64_t kWarmupBase = 1'000'000;
    for (std::uint64_t i = 0; i < kWarmup; ++i) {
        const Side s = (i % 2 == 0) ? Side::Bid : Side::Ask;
        const std::int64_t px = 9900 + static_cast<std::int64_t>((i % 200));
        engine.add_limit(OrderId{kWarmupBase + i}, s, Price{px}, Quantity{1});
    }
    std::cout << " done.\n";

    // Measured run: N scored ops, HdrHistogram-recorded.
    constexpr std::uint64_t kN = 100'000;
    constexpr std::uint64_t kMeasureBase = 2'000'000;

    hdr_histogram* hist = nullptr;
    // range: 1ns .. 1s, 3 sig-figs.
    hdr_init(1, 1'000'000'000, 3, &hist);

    std::cout << "Measuring " << kN << " scored add_limit ops..." << std::flush;
    for (std::uint64_t i = 0; i < kN; ++i) {
        const Side s = (i % 2 == 0) ? Side::Bid : Side::Ask;
        const std::int64_t px = 9900 + static_cast<std::int64_t>((i % 200));
        const auto t0 = hires::now();
        engine.add_limit(OrderId{kMeasureBase + i}, s, Price{px}, Quantity{1});
        const auto t1 = hires::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        hdr_record_value(hist, ns);
    }
    std::cout << " done.\n\n";

    const auto p50 = hdr_value_at_percentile(hist, 50.0);
    const auto p90 = hdr_value_at_percentile(hist, 90.0);
    const auto p99 = hdr_value_at_percentile(hist, 99.0);
    const auto p999 = hdr_value_at_percentile(hist, 99.9);
    const auto pmax = hdr_max(hist);

    auto print_line = [](const char* label, std::int64_t ns) {
        std::printf("  %-6s = %8lld ns (%6.2f us)\n", label,
                    static_cast<long long>(ns), static_cast<double>(ns) / 1000.0);
    };
    std::cout << "Latency over " << kN << " scored add_limit ops:\n";
    print_line("p50",  p50);
    print_line("p90",  p90);
    print_line("p99",  p99);
    print_line("p999", p999);
    print_line("max",  pmax);

    // Write docs/bench.md (overwrite each run).
    std::ofstream md(md_path);
    md << "# clob — scoring latency benchmark\n\n"
       << "Measured: scored `Engine::add_limit` (FeatureState snapshot + Scorer + "
       << "ScoreSink + match_against + FeatureState observe)\n"
       << "over **" << kN << "** ops after **" << kWarmup << "** warmup ops on a "
       << "1000-order seeded book.\n\n"
       << "Includes: feature assembly + ONNX inference + matcher hot path. "
       << "Excludes: IO, model load, process startup.\n\n"
       << "**SLO** (ADR 0001 amendment): p99 < 1ms. **Stretch** (ADR 0002 W14): p99 < 200us via TreeLite.\n\n"
       << "| Percentile | Latency (ns) | Latency (us) |\n"
       << "|---|---|---|\n";
    auto write_row = [&](const char* name, std::int64_t ns) {
        md << "| " << name << " | " << ns << " | "
           << (static_cast<double>(ns) / 1000.0) << " |\n";
    };
    write_row("p50", p50);
    write_row("p90", p90);
    write_row("p99", p99);
    write_row("p999", p999);
    write_row("max", pmax);

    md << "\nGate: ";
    md << (p99 < 1'000'000 ? ":white_check_mark: PASS" : ":x: FAIL");
    md << " (p99 = " << (static_cast<double>(p99) / 1000.0) << " us"
       << " vs SLO 1000 us)\n";

    hdr_close(hist);

    if (p99 >= 1'000'000) {
        std::cerr << "\nSLO BREACH: p99 = " << (p99 / 1000.0) << " us >= 1000 us\n";
        return 1;
    }
    std::cout << "\nSLO OK (p99 = " << (p99 / 1000.0) << " us < 1000 us)\n";
    return 0;
}
