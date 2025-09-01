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

