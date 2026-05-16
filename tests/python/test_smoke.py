import polars as pl


def test_polars_version() -> None:
    # Pinned >= 1.21 in pyproject.toml; verify at runtime.
    major, minor, *_ = pl.__version__.split(".")
    assert (int(major), int(minor)) >= (1, 21), f"polars too old: {pl.__version__}"


def test_polars_works() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    assert df["x"].sum() == 6
