// CI gate for the scoring latency SLO (ADR 0001 amendment).
//
// Runs the bench-scoring binary; FAILS the test if it exits non-zero,
// which it does when p99 >= 1ms. The binary writes docs/bench.md as a
// side effect, keeping the latency report committed alongside the gate.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

TEST_CASE("Scoring latency: p99 < 1ms (ADR 0001 amended SLO, plan v3 Q5)",
          "[bench][slo][integration]") {
    // Path injected at compile time via target_compile_definitions.
    const std::string cmd = std::string(CLOB_BENCH_SCORING_CLI);
    // Run from repo root so model/artifacts/ paths resolve relative to CWD.
    // The test binary lives at build/Debug/tests/integration/test_bench_slo;
    // chdir to ../../../../ would be repo root — but for portability the
    // bench binary takes explicit paths.
    // CLOB_REPO_ROOT is also injected.
    const std::string root = CLOB_REPO_ROOT;
    const std::string model = root + "/model/artifacts/model.onnx";
    const std::string lut   = root + "/model/artifacts/microprice_g.json";
    const std::string out   = root + "/docs/bench.md";

    if (!std::filesystem::exists(model) || !std::filesystem::exists(lut)) {
        SKIP("model.onnx + microprice_g.json missing — "
             "run `uv run python -m model.train` first");
    }

    const std::string full = cmd + " '" + model + "' '" + lut + "' '" + out + "'";
    const int rc = std::system(full.c_str());
    REQUIRE(rc == 0);  // bench binary exits non-zero on SLO breach
}
