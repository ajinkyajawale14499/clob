// clob_py — Python bindings for the C++ matching engine + scorer.
//
// W9 (Engine, Book, Fill, Side) + W10 (Scorer, MicropriceLut, scoring Engine ctor).

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/matching/engine.hpp"
#include "core/scoring/feature_state.hpp"
#include "core/scoring/scorer.hpp"

namespace py = pybind11;
using namespace clob;

namespace {

// Convert a numpy float32 array of length 19 into a ScoredFeatures.
ScoredFeatures features_from_array(py::array_t<float, py::array::c_style> arr) {
    auto buf = arr.request();
    if (buf.size != 19) {
        throw std::runtime_error(std::format(
            "Scorer.score: expected 19 features, got {}", buf.size));
    }
    ScoredFeatures f{};
    std::memcpy(&f, buf.ptr, 19 * sizeof(float));
    return f;
}

}  // namespace

PYBIND11_MODULE(clob_py, m) {
    m.doc() = "clob C++ matching engine + ONNX scorer — Python bindings";

    py::enum_<Side>(m, "Side")
        .value("Bid", Side::Bid)
        .value("Ask", Side::Ask)
        .export_values();

    py::class_<Fill>(m, "Fill")
        .def_property_readonly("taker_id",
            [](const Fill& f) { return f.taker_id.value(); })
        .def_property_readonly("maker_id",
            [](const Fill& f) { return f.maker_id.value(); })
        .def_property_readonly("price",
            [](const Fill& f) { return f.price.value(); })
        .def_property_readonly("quantity",
            [](const Fill& f) { return f.quantity.value(); })
        .def("__repr__", [](const Fill& f) {
            return std::format("Fill(taker={}, maker={}, price={}, qty={})",
                                f.taker_id.value(), f.maker_id.value(),
                                f.price.value(), f.quantity.value());
        });

    py::class_<Book>(m, "Book")
        .def("best_bid", [](const Book& b) -> std::optional<std::int64_t> {
            auto p = b.best_bid();
            return p ? std::optional{p->value()} : std::nullopt;
        })
        .def("best_ask", [](const Book& b) -> std::optional<std::int64_t> {
            auto p = b.best_ask();
            return p ? std::optional{p->value()} : std::nullopt;
        })
        .def("level_quantity",
             [](const Book& b, Side s, std::int64_t p) -> std::int64_t {
                 const Level* lvl = b.level_at(s, Price{p});
                 return lvl ? lvl->total_quantity().value() : 0;
             },
             py::arg("side"), py::arg("price"));

    // ---- W10: MicropriceLut + Scorer ----------------------------------------

    py::class_<MicropriceLut, std::shared_ptr<MicropriceLut>>(m, "MicropriceLut")
        .def_static("load",
                    [](const std::string& path) {
                        return std::make_shared<MicropriceLut>(
                            MicropriceLut::load(path));
                    },
                    py::arg("json_path"),
                    "Load a Stoikov G(I,S) lookup table from JSON.")
        .def("lookup",
             [](const MicropriceLut& l, double imb, std::int64_t sp) {
                 return l.lookup(imb, sp);
             });

    py::class_<Scorer, std::shared_ptr<Scorer>>(m, "Scorer")
        .def_static("load",
                    [](const std::string& path) {
                        return std::make_shared<Scorer>(path);
                    },
                    py::arg("onnx_path"),
                    "Load a LightGBM-exported ONNX model (multiclass, zipmap=False).")
        .def("score",
             [](Scorer& s, py::array_t<float, py::array::c_style> arr) -> double {
                 return s.score(features_from_array(arr));
             },
             py::arg("features"),
             "Score a single 19-feature vector; returns P(Up) - P(Down) ∈ [-1,1].")
        .def("score_batch",
             [](Scorer& s, py::array_t<float, py::array::c_style> arr) -> py::array_t<double> {
                 auto buf = arr.request();
                 if (buf.ndim != 2 || buf.shape[1] != 19) {
                     throw std::runtime_error("score_batch: expected (N, 19) array");
                 }
                 const auto n = static_cast<std::size_t>(buf.shape[0]);
                 std::vector<ScoredFeatures> batch(n);
                 const float* ptr = static_cast<const float*>(buf.ptr);
                 for (std::size_t i = 0; i < n; ++i) {
                     std::memcpy(&batch[i], ptr + i * 19, 19 * sizeof(float));
                 }
                 auto scores = s.score_batch(batch);
                 return py::array_t<double>(static_cast<py::ssize_t>(n), scores.data());
             })
        .def("probs_batch",
             [](Scorer& s, py::array_t<float, py::array::c_style> arr) -> py::array_t<double> {
                 auto buf = arr.request();
                 if (buf.ndim != 2 || buf.shape[1] != 19) {
                     throw std::runtime_error("probs_batch: expected (N, 19) array");
                 }
                 const auto n = static_cast<std::size_t>(buf.shape[0]);
                 std::vector<ScoredFeatures> batch(n);
                 const float* ptr = static_cast<const float*>(buf.ptr);
                 for (std::size_t i = 0; i < n; ++i) {
                     std::memcpy(&batch[i], ptr + i * 19, 19 * sizeof(float));
                 }
                 auto probs = s.probs_batch(batch);
                 std::vector<double> flat(n * 3);
                 for (std::size_t i = 0; i < n; ++i) {
                     for (std::size_t c = 0; c < 3; ++c) flat[i * 3 + c] = probs[i][c];
                 }
                 std::array<py::ssize_t, 2> shape{static_cast<py::ssize_t>(n), 3};
                 return py::array_t<double>(shape, flat.data());
             });

    // ---- Engine -------------------------------------------------------------

    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        // W10 scoring ctor — keyword args.
        .def(py::init([](std::shared_ptr<Scorer> scorer,
                          std::optional<py::function> score_sink,
                          const std::string& ticker,
                          std::shared_ptr<MicropriceLut> lut) {
                 Engine::ScoreSink sink;
                 if (score_sink && !score_sink->is_none()) {
                     auto py_cb = *score_sink;
                     sink = [py_cb](OrderId id, double s) {
                         py::gil_scoped_acquire gil;
                         py_cb(id.value(), s);
                     };
                 }
                 return std::make_unique<Engine>(
                     nullptr, scorer.get(), std::move(sink), ticker, lut.get());
             }),
             py::arg("scorer"), py::arg("score_sink") = py::none(),
             py::arg("ticker") = "", py::arg("lut") = nullptr,
             "W10 ctor: enables scoring path. scorer + lut must outlive Engine.")
        .def("add_limit",
             [](Engine& e, std::uint64_t id, Side s, std::int64_t p, std::int64_t q) {
                 return e.add_limit(OrderId{id}, s, Price{p}, Quantity{q});
             },
             py::arg("order_id"), py::arg("side"), py::arg("price"), py::arg("qty"))
        .def("add_market",
             [](Engine& e, std::uint64_t id, Side s, std::int64_t q) {
                 return e.add_market(OrderId{id}, s, Quantity{q});
             },
             py::arg("order_id"), py::arg("side"), py::arg("qty"))
        .def("add_ioc",
             [](Engine& e, std::uint64_t id, Side s, std::int64_t p, std::int64_t q) {
                 return e.add_ioc(OrderId{id}, s, Price{p}, Quantity{q});
             },
             py::arg("order_id"), py::arg("side"), py::arg("price"), py::arg("qty"))
        .def("cancel",
             [](Engine& e, std::uint64_t id) {
                 return e.cancel(OrderId{id});
             },
             py::arg("order_id"))
        .def("cancel_replace",
             [](Engine& e, std::uint64_t old_id, std::uint64_t new_id,
                std::int64_t p, std::int64_t q) {
                 return e.cancel_replace(OrderId{old_id}, OrderId{new_id},
                                          Price{p}, Quantity{q});
             },
             py::arg("old_id"), py::arg("new_id"), py::arg("price"), py::arg("qty"))
        .def("book", &Engine::book, py::return_value_policy::reference_internal);
}
