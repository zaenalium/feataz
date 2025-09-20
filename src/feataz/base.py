from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import polars as pl


def _ensure_polars_df(df: pl.DataFrame) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df

    # Lazy import so pandas remains optional
    pd = None
    if df.__class__.__module__.startswith("pandas"):
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise TypeError(
                "Pandas support requires installing pandas; install pandas to pass pandas.DataFrame"
            ) from exc
    if pd is not None and isinstance(df, pd.DataFrame):  # type: ignore[name-defined]
        try:
            return pl.from_pandas(df)
        except (ImportError, ModuleNotFoundError):
            # Fallback without pyarrow: construct via Python lists
            data = {col: df[col].tolist() for col in df.columns}
            return pl.DataFrame(data)

    raise TypeError("Expected a polars.DataFrame or pandas.DataFrame")


def _ensure_polars_series(series: pl.Series) -> pl.Series:
    if isinstance(series, pl.Series):
        return series

    pd = None
    if series.__class__.__module__.startswith("pandas"):
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise TypeError(
                "Pandas support requires installing pandas; install pandas to pass pandas.Series"
            ) from exc
    if pd is not None and isinstance(series, pd.Series):  # type: ignore[name-defined]
        name = series.name if series.name is not None else "column_0"
        try:
            return pl.from_pandas(series.to_frame(name=name)).get_column(name)
        except (ImportError, ModuleNotFoundError):
            # Fallback: build from list without pyarrow
            return pl.Series(name, series.tolist())

    raise TypeError("Expected a polars.Series or pandas.Series")


def _as_list(x: Optional[Sequence[str]], fallback: List[str]) -> List[str]:
    if x is None:
        return list(fallback)
    return list(x)


class Transformer:
    """Simple fit/transform interface for Polars DataFrames.

    Subclasses should implement fit(self, df: pl.DataFrame) -> "Transformer"
    and transform(self, df: pl.DataFrame) -> pl.DataFrame.
    """

    feature_names_in_: List[str] | None = None
    is_fitted_: bool = False

    def fit(self, df: pl.DataFrame) -> "Transformer":  # pragma: no cover
        raise NotImplementedError

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.fit(df).transform(df)

    # Optional helpers
    def get_feature_names_out(self) -> List[str]:
        names = getattr(self, "feature_names_out_", None)
        if names is None:
            # Not all transformers track this; return empty list by default
            return []
        return list(names)

    def to_dict(self) -> dict:
        # Best-effort serialization of learned state (attrs ending with '_')
        state = {}
        for k, v in self.__dict__.items():
            if not k.endswith("_"):
                continue
            # skip large objects like sklearn models by type name
            tname = type(v).__name__
            if tname in {"DecisionTreeClassifier", "DecisionTreeRegressor"}:
                continue
            try:
                pl.Series([v])  # quick serializability probe for simple values
                state[k] = v
            except Exception:
                # try converting polars to dict
                if isinstance(v, pl.DataFrame):
                    state[k] = {"__type__": "pldf", "columns": v.columns, "rows": v.rows()}
                elif isinstance(v, (list, dict, str, int, float, type(None))):
                    state[k] = v
                else:
                    # skip
                    pass
        state["__class__"] = self.__class__.__name__
        return state

    def from_dict(self, state: dict) -> "Transformer":
        # Restore attrs saved by to_dict
        for k, v in state.items():
            if k == "__class__":
                continue
            if isinstance(v, dict) and v.get("__type__") == "pldf":
                df = pl.DataFrame(v["rows"], schema=v["columns"])  # type: ignore[arg-type]
                setattr(self, k, df)
            else:
                setattr(self, k, v)
        return self
