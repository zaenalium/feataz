from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import polars as pl


def _ensure_polars_df(df: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(df, pl.DataFrame):
        raise TypeError("Expected a polars.DataFrame")
    return df


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
