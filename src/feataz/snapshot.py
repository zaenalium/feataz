from __future__ import annotations

from typing import List, Optional, Sequence

import polars as pl

from .base import Transformer, _ensure_polars_df


def _infer_numeric(df: pl.DataFrame, exclude: Sequence[str] = ()) -> List[str]:
    cols: List[str] = []
    exclude_set = set(exclude)
    for n, t in zip(df.columns, df.dtypes):
        if n in exclude_set:
            continue
        if pl.datatypes.is_numeric(t):
            cols.append(n)
    return cols


def _infer_categorical(df: pl.DataFrame, exclude: Sequence[str] = ()) -> List[str]:
    cols: List[str] = []
    exclude_set = set(exclude)
    for n, t in zip(df.columns, df.dtypes):
        if n in exclude_set:
            continue
        if pl.datatypes.is_string_dtype(t) or t == pl.Categorical:
            cols.append(n)
    return cols


def _normalize_groupby(
    df: pl.DataFrame, groupby: Optional[Sequence[str] | Sequence[Sequence[str]]], time_col: str
) -> List[List[str]]:
    # Default: each categorical column (excluding time) is its own grouping set
    if groupby is None:
        return [[c] for c in _infer_categorical(df, exclude=(time_col,))]
    if len(groupby) > 0 and isinstance(groupby[0], str):  # type: ignore[index]
        return [list(groupby)]  # type: ignore[arg-type]
    return [list(gs) for gs in groupby]  # type: ignore[list-item]


class TimeSnapshotAggregator(Transformer):
    """Per-row trailing window aggregations over time, optionally grouped.

    - Computes features like sum/mean/count of value columns over the last X
      days/weeks/months/quarters/years relative to each row's timestamp.
    - Uses Polars `group_by_rolling` with a time-based window.

    Parameters
    - time_column: Name of the Date/Datetime column to use as the time index.
    - groupby: list[str] or list[list[str]] group sets. Default: each categorical column separately.
    - value_columns: numeric columns to aggregate. Default: infer numeric (excluding time & groups).
    - windows: list of duration strings like '7d', '4w', '1mo', '1q', '1y'.
    - aggregations: subset of {'sum','mean','min','max','median','std','var','count','n_unique'}.
    - include_current: include the current row in the trailing window (default False).
    - name_sep/by_token/group_sep: naming tokens.
    - drop_original: drop original columns after adding features (default False).
    """

    def __init__(
        self,
        time_column: str,
        groupby: Optional[Sequence[str] | Sequence[Sequence[str]]] = None,
        value_columns: Optional[Sequence[str]] = None,
        windows: Sequence[str] = ("7d",),
        aggregations: Sequence[str] = ("sum",),
        include_current: bool = False,
        drop_original: bool = False,
        name_sep: str = "__",
        by_token: str = "by",
        group_sep: str = "__and__",
    ) -> None:
        self.time_column = time_column
        self.groupby = groupby
        self.value_columns = None if value_columns is None else list(value_columns)
        self.windows = list(windows)
        self.aggregations = list(aggregations)
        self.include_current = include_current
        self.drop_original = drop_original
        self.name_sep = name_sep
        self.by_token = by_token
        self.group_sep = group_sep

        self.groupby_sets_: List[List[str]] = []
        self.value_columns_: List[str] = []
        self.aggregations_: List[str] = []

    def fit(self, df: pl.DataFrame) -> "TimeSnapshotAggregator":
        df = _ensure_polars_df(df)
        if self.time_column not in df.columns:
            raise ValueError(f"time column '{self.time_column}' not found")
        dtype = df.get_column(self.time_column).dtype
        if dtype not in (pl.Date, pl.Datetime):
            raise TypeError(
                f"time column '{self.time_column}' must be Date or Datetime, got {dtype}"
            )
        self.groupby_sets_ = _normalize_groupby(df, self.groupby, self.time_column)
        # determine value columns: infer numeric excluding time & group columns
        exclude = {self.time_column} | {c for gs in self.groupby_sets_ for c in gs}
        self.value_columns_ = _infer_numeric(df, exclude=tuple(exclude)) if self.value_columns is None else list(self.value_columns)
        for v in self.value_columns_:
            if v not in df.columns:
                raise ValueError(f"Value column '{v}' not found")
        # validate aggregations
        valid_aggs = {"sum", "mean", "min", "max", "median", "std", "var", "count", "n_unique"}
        for a in self.aggregations:
            if a not in valid_aggs:
                raise ValueError(f"Unsupported aggregation '{a}'. Valid: {sorted(valid_aggs)}")
        self.aggregations_ = list(self.aggregations)
        self.feature_names_in_ = [self.time_column] + list(exclude) + self.value_columns_
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
        closed = "both" if self.include_current else "left"
        time_col = self.time_column

        for gs in self.groupby_sets_:
            # Sort by group keys and time for rolling groups
            work = df.sort(gs + [time_col]) if gs else df.sort(time_col)
            for w in self.windows:
                exprs: List[pl.Expr] = []
                key_str = self.group_sep.join(gs) if gs else "_"
                for v in self.value_columns_:
                    for a in self.aggregations_:
                        out_name = self.name_sep.join([v, a, f"last_{w}", self.by_token, key_str])
                        exprs.append(self._agg_expr(v, a, out_name))
                if not exprs:
                    continue
                if gs:
                    roll = work.group_by_rolling(index_column=time_col, period=w, by=gs, closed=closed).agg(exprs)
                else:
                    roll = work.group_by_rolling(index_column=time_col, period=w, closed=closed).agg(exprs)
                # join back on group keys + time
                join_keys = gs + [time_col]
                out = out.join(roll, on=join_keys, how="left")

        if self.drop_original:
            keep_cols = [c for c in out.columns if c not in set(self.feature_names_in_ or [])]
            out = out.select(keep_cols)
        return out
