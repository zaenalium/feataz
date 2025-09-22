import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

pl = pytest.importorskip("polars", reason="polars is required for feataz tests")
sklearn_datasets = pytest.importorskip(
    "sklearn.datasets", reason="scikit-learn is required for the wine dataset"
)

if not hasattr(pl.datatypes, "is_numeric"):
    def _is_numeric(dtype: pl.DataType) -> bool:  # type: ignore[name-defined]
        try:
            return bool(dtype.is_numeric())  # type: ignore[attr-defined]
        except AttributeError:
            return False

    pl.datatypes.is_numeric = _is_numeric  # type: ignore[attr-defined]

if not hasattr(pl.datatypes, "is_string_dtype"):
    def _is_string(dtype: pl.DataType) -> bool:  # type: ignore[name-defined]
        try:
            return bool(dtype.is_(pl.Utf8) or dtype.is_(pl.String))  # type: ignore[attr-defined]
        except AttributeError:
            return False

    pl.datatypes.is_string_dtype = _is_string  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def wine_df() -> pl.DataFrame:
    loader = sklearn_datasets.load_wine()
    data = {name: loader.data[:, idx].astype(float) for idx, name in enumerate(loader.feature_names)}
    data["target"] = loader.target.astype(int)
    df = pl.DataFrame(data)
    # Attach helper columns used by multiple tests
    return df.with_columns(
        pl.col("target").cast(pl.Utf8).map_elements(lambda x: f"class_{x}", return_dtype=pl.Utf8).alias("target_str"),
        pl.Series("index", range(df.height)),
    )


@pytest.fixture()
def wine_binary(wine_df: pl.DataFrame) -> pl.DataFrame:
    return wine_df.with_columns(
        (pl.col("target") == 0).cast(pl.Int64).alias("target_bin")
    )


@pytest.fixture()
def wine_with_time(wine_df: pl.DataFrame) -> pl.DataFrame:
    # Build a deterministic timestamp column for aggregators
    start = dt.datetime(2020, 1, 1)
    timestamps = [start + dt.timedelta(days=int(i)) for i in range(wine_df.height)]
    return wine_df.with_columns(pl.Series("ts", timestamps))


@pytest.fixture()
def wine_w_target(wine_with_time: pl.DataFrame) -> pl.DataFrame:
    df = wine_with_time
    return df.with_columns(
        (pl.col("target") == 0).cast(pl.Int64).alias("target_bin")
    )


@pytest.fixture()
def positive_numeric_columns(wine_df: pl.DataFrame) -> list[str]:
    return [
        name
        for name in wine_df.columns
        if wine_df.get_column(name).cast(pl.Float64).min() > 0 and name not in {"index", "target"}
    ]
