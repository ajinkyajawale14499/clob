#include "core/scoring/scorer.hpp"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace clob {

namespace {

constexpr const char* kInputName = "input";
constexpr std::size_t kNumFeatures = 19;
constexpr std::size_t kNumClasses = 3;

}  // namespace

Scorer::Scorer(const std::filesystem::path& onnx_path)
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "clob_scorer")),
      session_() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), opts);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("Scorer: failed to load ") +
                                  onnx_path.string() + ": " + e.what());
    }
}

Scorer::~Scorer() = default;
Scorer::Scorer(Scorer&&) noexcept = default;
Scorer& Scorer::operator=(Scorer&&) noexcept = default;

double Scorer::score(const ScoredFeatures& f) {
    auto probs = probs_batch({f});
    // Returns P(Up) - P(Down).
    return probs[0][2] - probs[0][0];
}

std::vector<double> Scorer::score_batch(const std::vector<ScoredFeatures>& batch) {
    auto probs = probs_batch(batch);
    std::vector<double> out;
    out.reserve(probs.size());
    for (const auto& p : probs) out.push_back(p[2] - p[0]);
    return out;
}

std::vector<std::array<double, 3>> Scorer::probs_batch(
    const std::vector<ScoredFeatures>& batch) {
    if (batch.empty()) return {};

    const std::size_t n = batch.size();
    // Pack batch into a contiguous float buffer of shape (N, 19).
    std::vector<float> flat(n * kNumFeatures);
    for (std::size_t i = 0; i < n; ++i) {
        std::array<float, kNumFeatures> row{};
        batch[i].to_array(row);
        for (std::size_t k = 0; k < kNumFeatures; ++k) {
            flat[i * kNumFeatures + k] = row[k];
        }
    }

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> shape = {static_cast<int64_t>(n),
                                     static_cast<int64_t>(kNumFeatures)};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem, flat.data(), flat.size(), shape.data(), shape.size());

    const char* input_names[] = {kInputName};
    // The exported model has two outputs: "label" (int64) and "probabilities".
    // Query the names dynamically to avoid hardcoding the model's output name.
    Ort::AllocatorWithDefaultOptions alloc;
    auto out0_name = session_->GetOutputNameAllocated(0, alloc);
    auto out1_name = session_->GetOutputNameAllocated(1, alloc);
    const char* output_names[] = {out0_name.get(), out1_name.get()};

    Ort::RunOptions ro;
    auto outputs = session_->Run(ro, input_names, &input_tensor, 1,
                                  output_names, 2);

    // outputs[1] is shape (N, 3) float — plain tensor since zipmap=False at export.
    const float* probs_ptr = outputs[1].GetTensorData<float>();
    std::vector<std::array<double, 3>> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < kNumClasses; ++c) {
            result[i][c] = static_cast<double>(probs_ptr[i * kNumClasses + c]);
        }
    }
    return result;
}

}  // namespace clob
