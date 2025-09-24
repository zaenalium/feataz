from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import polars as pl

from .base import Transformer, _ensure_polars_df


def _infer_numeric(df: pl.DataFrame) -> List[str]:
    cols: List[str] = []
    for n, t in zip(df.columns, df.dtypes):
        if t.is_numeric():
            cols.append(n)
    return cols


def _infer_categorical(df: pl.DataFrame) -> List[str]:
    cols: List[str] = []
    for n, t in zip(df.columns, df.dtypes):
        if t == pl.String or t == pl.Categorical:
            cols.append(n)
    return cols


def _normalize_groupby(
    df: pl.DataFrame, groupby: Optional[Sequence[str] | Sequence[Sequence[str]]] | None
) -> List[List[str]]:
    # None => use each categorical column as its own grouping set
    if groupby is None:
        return [[c] for c in _infer_categorical(df)]
    # Sequence[str] => single set
    if len(groupby) > 0 and isinstance(groupby[0], str):  # type: ignore[index]
        return [list(groupby)]  # type: ignore[arg-type]
    # Sequence[Sequence[str]]
    return [list(gs) for gs in groupby]  # type: ignore[list-item]


class FeatureInteractions(Transformer):
    """Group-based aggregations joined back to the DataFrame.

    Examples:
    - sum of numeric columns grouped by categorical columns
    - mean/min/max/std per group
    - multiple grouping variable sets supported
    """

    def __init__(
        self,
        groupby: Optional[Sequence[str] | Sequence[Sequence[str]]] = None,
        value_columns: Optional[Sequence[str]] = None,
        aggregations: Sequence[str] = ("sum", "mean"),
        drop_original: bool = False,
        name_sep: str = "__",
        by_token: str = "by",
        group_sep: str = "__and__",
    ) -> None:
        self.groupby = groupby
        self.value_columns = None if value_columns is None else list(value_columns)
        self.aggregations = list(aggregations)
        self.drop_original = drop_original
        self.name_sep = name_sep
        self.by_token = by_token
        self.group_sep = group_sep

        self.groupby_sets_: List[List[str]] = []
        self.value_columns_: List[str] = []
        self.aggregations_: List[str] = []

    def fit(self, df: pl.DataFrame) -> "FeatureInteractions":
        df = _ensure_polars_df(df)
        self.groupby_sets_ = _normalize_groupby(df, self.groupby)
        self.value_columns_ = (
            _infer_numeric(df) if self.value_columns is None else list(self.value_columns)
        )
        # validate that all group cols exist
        for gs in self.groupby_sets_:
            for g in gs:
                if g not in df.columns:
                    raise ValueError(f"Group column '{g}' not found in DataFrame")
        # validate that value cols exist
        for v in self.value_columns_:
            if v not in df.columns:
                raise ValueError(f"Value column '{v}' not found in DataFrame")
        # normalize aggregations
        valid_aggs = {"sum", "mean", "min", "max", "median", "std", "var", "count", "n_unique"}
        for a in self.aggregations:
            if a not in valid_aggs:
                raise ValueError(f"Unsupported aggregation '{a}'. Valid: {sorted(valid_aggs)}")
        self.aggregations_ = list(self.aggregations)
        self.feature_names_in_ = list({c for gs in self.groupby_sets_ for c in gs} | set(self.value_columns_))
        self.is_fitted_ = True
        return self

    def _agg_expr(self, value_col: str, agg: str, out_name: str) -> pl.Expr:
        if agg == "sum":
            return pl.col(value_col).sum().alias(out_name)
        if agg == "mean":
            return pl.col(value_col).mean().alias(out_name)
        if agg == "min":
            return pl.col(value_col).min().alias(out_name)
        if agg == "max":
            return pl.col(value_col).max().alias(out_name)
        if agg == "median":
            return pl.col(value_col).median().alias(out_name)
        if agg == "std":
            return pl.col(value_col).std().alias(out_name)
        if agg == "var":
            return pl.col(value_col).var().alias(out_name)
        if agg == "n_unique":
            return pl.col(value_col).n_unique().alias(out_name)
        if agg == "count":
            return pl.len().alias(out_name)
        raise ValueError(f"Unsupported aggregation '{agg}'")

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for gs in self.groupby_sets_:
            exprs: List[pl.Expr] = []
            key_str = self.group_sep.join(gs)
            for v in self.value_columns_:
                for a in self.aggregations_:
                    out_name = self.name_sep.join([v, a, self.by_token, key_str])
                    exprs.append(self._agg_expr(v, a, out_name))
            if not exprs:
                continue
            agg_df = df.group_by(gs).agg(exprs)
            out = out.join(agg_df, on=gs, how="left")
        if self.drop_original:
            keep_cols = [c for c in out.columns if c not in set(self.feature_names_in_ or [])]
            out = out.select(keep_cols)
        return out
