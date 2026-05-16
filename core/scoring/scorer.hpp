#pragma once

// Scorer — wraps Ort::Session for the LightGBM-exported ONNX model.
//
// The model expects float32 input of shape (N, 19) and emits:
//   output[0]: int64 label tensor (N,)         — argmax class
//   output[1]: float prob tensor   (N, 3)      — [P(Down), P(Stable), P(Up)]
// (zipmap=False was used at export time; see model/train.py:export_to_onnx).
//
// score(features) -> P(Up) - P(Down) ∈ [-1, +1]; positive = expects up move.

#include <filesystem>
#include <memory>
#include <vector>

#include "core/scoring/feature_state.hpp"

namespace Ort {
class Env;
class Session;
class AllocatorWithDefaultOptions;
}

namespace clob {

class Scorer {
public:
    explicit Scorer(const std::filesystem::path& onnx_path);
    ~Scorer();

    Scorer(const Scorer&) = delete;
    Scorer& operator=(const Scorer&) = delete;
    Scorer(Scorer&&) noexcept;
    Scorer& operator=(Scorer&&) noexcept;

    // Score a single feature vector. Returns P(Up) - P(Down).
    [[nodiscard]] double score(const ScoredFeatures& f);

    // Score a batch — returns one score per row, in input order.
    [[nodiscard]] std::vector<double> score_batch(
        const std::vector<ScoredFeatures>& batch);

    // Lower-level: returns the raw 3-class probabilities P(Down|Stable|Up)
    // for tests that want exact LightGBM-vs-Ort parity inspection.
    [[nodiscard]] std::vector<std::array<double, 3>> probs_batch(
        const std::vector<ScoredFeatures>& batch);

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

}  // namespace clob
