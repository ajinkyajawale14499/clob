"""Smoke tests for the pybind11 Engine wrapper (clob_py)."""

import pytest

try:
    import clob_py
except ImportError:
    pytest.skip(
        "clob_py not built — run `cmake --preset default && "
        "cmake --build build/Release --parallel` first",
        allow_module_level=True,
    )


def test_engine_construct():
    e = clob_py.Engine()
    assert e.book().best_bid() is None
    assert e.book().best_ask() is None


def test_engine_add_limit_returns_fills():
    e = clob_py.Engine()
    e.add_limit(1, clob_py.Side.Ask, 10000, 5)
    fills = e.add_limit(2, clob_py.Side.Bid, 10000, 3)
    assert len(fills) == 1
    f = fills[0]
    assert f.taker_id == 2
    assert f.maker_id == 1
    assert f.price == 10000
    assert f.quantity == 3


def test_engine_book_inspection():
    e = clob_py.Engine()
    e.add_limit(1, clob_py.Side.Bid, 99, 5)
    e.add_limit(2, clob_py.Side.Ask, 101, 5)
    assert e.book().best_bid() == 99
    assert e.book().best_ask() == 101


def test_engine_add_market():
    e = clob_py.Engine()
    e.add_limit(1, clob_py.Side.Ask, 100, 5)
    fills = e.add_market(2, clob_py.Side.Bid, 3)
    assert len(fills) == 1
    assert fills[0].quantity == 3


def test_engine_add_ioc():
    e = clob_py.Engine()
    e.add_limit(1, clob_py.Side.Ask, 100, 3)
    fills = e.add_ioc(2, clob_py.Side.Bid, 100, 10)
    assert len(fills) == 1
    assert fills[0].quantity == 3
    # 7 dropped, not rested
    assert e.book().best_bid() is None


def test_engine_cancel():
    e = clob_py.Engine()
    e.add_limit(1, clob_py.Side.Bid, 100, 5)
    assert e.cancel(1) is True
    assert e.book().best_bid() is None
    assert e.cancel(999) is False


def test_engine_cancel_replace():
    e = clob_py.Engine()
    e.add_limit(1, clob_py.Side.Bid, 100, 5)
    e.add_limit(2, clob_py.Side.Bid, 100, 5)
    fills = e.cancel_replace(1, 3, 100, 5)
    assert fills == []  # no cross
    # The replacement is at the back of the queue at 100.


def test_fill_repr():
    e = clob_py.Engine()
    e.add_limit(1, clob_py.Side.Ask, 100, 5)
    fills = e.add_limit(2, clob_py.Side.Bid, 100, 3)
    r = repr(fills[0])
    assert "Fill" in r
    assert "taker=2" in r
    assert "maker=1" in r
