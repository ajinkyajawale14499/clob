#include "core/scoring/scorer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

using namespace clob;

namespace {

std::filesystem::path repo_root() {
    // tests/unit/test_scorer -> ../../  is repo root
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
}

bool model_artifact_exists() {
    return std::filesystem::exists(repo_root() / "model" / "artifacts" / "model.onnx");
}

}  // namespace

TEST_CASE("Scorer: throws on missing file", "[scorer]") {
    REQUIRE_THROWS(Scorer("/definitely/does/not/exist.onnx"));
}

TEST_CASE("Scorer: loads real model + scores in [-1, +1]", "[scorer]") {
    if (!model_artifact_exists()) {
        SKIP("model/artifacts/model.onnx not present — run `uv run python -m model.train` first");
    }
    Scorer s(repo_root() / "model" / "artifacts" / "model.onnx");
    ScoredFeatures f{};
    f.ticker_AAPL = 1.0f;
    f.is_warm_50 = 1.0f;
    f.is_warm_200 = 1.0f;
    f.spread_ticks = 1.0f;
    const double score = s.score(f);
    REQUIRE(score >= -1.0);
    REQUIRE(score <= 1.0);
}

TEST_CASE("Scorer: probs sum to 1.0", "[scorer]") {
    if (!model_artifact_exists()) {
        SKIP("model.onnx not present");
    }
    Scorer s(repo_root() / "model" / "artifacts" / "model.onnx");
    ScoredFeatures f{};
    f.ticker_AAPL = 1.0f;
    f.is_warm_50 = 1.0f;
    f.is_warm_200 = 1.0f;
    auto probs = s.probs_batch({f});
    REQUIRE(probs.size() == 1);
    const double total = probs[0][0] + probs[0][1] + probs[0][2];
    REQUIRE(std::abs(total - 1.0) < 1e-5);
}

TEST_CASE("Scorer: extreme bid imbalance pushes score positive", "[scorer]") {
    if (!model_artifact_exists()) {
        SKIP("model.onnx not present");
    }
    Scorer s(repo_root() / "model" / "artifacts" / "model.onnx");
    ScoredFeatures up{};
    up.ticker_AAPL = 1.0f; up.imbalance_l1 = 0.9f; up.is_warm_50 = 1.0f; up.is_warm_200 = 1.0f;
    ScoredFeatures down{};
    down.ticker_AAPL = 1.0f; down.imbalance_l1 = -0.9f; down.is_warm_50 = 1.0f; down.is_warm_200 = 1.0f;
    REQUIRE(s.score(up) > s.score(down));
}

TEST_CASE("Scorer: score_batch == loop of score", "[scorer]") {
    if (!model_artifact_exists()) {
        SKIP("model.onnx not present");
    }
    Scorer s(repo_root() / "model" / "artifacts" / "model.onnx");
    std::vector<ScoredFeatures> batch(10);
    for (std::size_t i = 0; i < batch.size(); ++i) {
        batch[i].ticker_AAPL = 1.0f;
        batch[i].imbalance_l1 = -1.0f + 0.2f * static_cast<float>(i);
        batch[i].is_warm_50 = 1.0f;
        batch[i].is_warm_200 = 1.0f;
    }
    auto batched = s.score_batch(batch);
    REQUIRE(batched.size() == 10);
    for (std::size_t i = 0; i < batch.size(); ++i) {
        const double single = s.score(batch[i]);
        REQUIRE(std::abs(batched[i] - single) < 1e-9);
    }
}
