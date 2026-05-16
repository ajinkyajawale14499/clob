// clob_py — Python bindings for the C++ matching engine.
//
// W9 minimal surface (Scorer added in W10):
//   Side (enum), Fill (struct), Book (read-only), Engine (mutating ops).
//
// Built via the bindings/CMakeLists.txt subdirectory; loaded into Python via
// PYTHONPATH that includes the CMake build dir (see tests/python/conftest.py).
// For W15 v1.0 polish, this gets re-packaged via scikit-build-core for
// `pip install -e .` ergonomics.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <format>
#include <optional>

#include "core/matching/engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(clob_py, m) {
    m.doc() = "clob C++ matching engine — Python bindings (W9 minimal surface)";

    py::enum_<clob::Side>(m, "Side")
        .value("Bid", clob::Side::Bid)
        .value("Ask", clob::Side::Ask)
        .export_values();

    py::class_<clob::Fill>(m, "Fill")
        .def_property_readonly("taker_id",
            [](const clob::Fill& f) { return f.taker_id.value(); })
        .def_property_readonly("maker_id",
            [](const clob::Fill& f) { return f.maker_id.value(); })
        .def_property_readonly("price",
            [](const clob::Fill& f) { return f.price.value(); })
        .def_property_readonly("quantity",
            [](const clob::Fill& f) { return f.quantity.value(); })
        .def("__repr__", [](const clob::Fill& f) {
            return std::format("Fill(taker={}, maker={}, price={}, qty={})",
                                f.taker_id.value(), f.maker_id.value(),
                                f.price.value(), f.quantity.value());
        });

    py::class_<clob::Book>(m, "Book")
        .def("best_bid", [](const clob::Book& b) -> std::optional<std::int64_t> {
            auto p = b.best_bid();
            return p ? std::optional{p->value()} : std::nullopt;
        })
        .def("best_ask", [](const clob::Book& b) -> std::optional<std::int64_t> {
            auto p = b.best_ask();
            return p ? std::optional{p->value()} : std::nullopt;
        });

    py::class_<clob::Engine>(m, "Engine")
        .def(py::init<>())
        .def("add_limit",
             [](clob::Engine& e, std::uint64_t id, clob::Side s,
                std::int64_t p, std::int64_t q) {
                 return e.add_limit(clob::OrderId{id}, s,
                                     clob::Price{p}, clob::Quantity{q});
             },
             py::arg("order_id"), py::arg("side"), py::arg("price"), py::arg("qty"))
        .def("add_market",
             [](clob::Engine& e, std::uint64_t id, clob::Side s, std::int64_t q) {
                 return e.add_market(clob::OrderId{id}, s, clob::Quantity{q});
             },
             py::arg("order_id"), py::arg("side"), py::arg("qty"))
        .def("add_ioc",
             [](clob::Engine& e, std::uint64_t id, clob::Side s,
                std::int64_t p, std::int64_t q) {
                 return e.add_ioc(clob::OrderId{id}, s,
                                   clob::Price{p}, clob::Quantity{q});
             },
             py::arg("order_id"), py::arg("side"), py::arg("price"), py::arg("qty"))
        .def("cancel",
             [](clob::Engine& e, std::uint64_t id) {
                 return e.cancel(clob::OrderId{id});
             },
             py::arg("order_id"))
        .def("cancel_replace",
             [](clob::Engine& e, std::uint64_t old_id, std::uint64_t new_id,
                std::int64_t p, std::int64_t q) {
                 return e.cancel_replace(clob::OrderId{old_id},
                                          clob::OrderId{new_id},
                                          clob::Price{p}, clob::Quantity{q});
             },
             py::arg("old_id"), py::arg("new_id"), py::arg("price"), py::arg("qty"))
        .def("book", &clob::Engine::book, py::return_value_policy::reference_internal);
}
