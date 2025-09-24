from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import polars as pl

from .base import Transformer, _ensure_polars_df


def _infer_numeric(df: pl.DataFrame, columns: Optional[Sequence[str]]) -> List[str]:
    if columns is not None:
        return list(columns)
    cols: List[str] = []
    for n, t in zip(df.columns, df.dtypes):
        if t.is_numeric():
            cols.append(n)
    return cols


class RobustScaler(Transformer):
    """Median/MAD scaling per column.

    x_scaled = (x - median) / (MAD * 1.4826)
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        drop_original: bool = True,
        suffix: str = "__rsc",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.drop_original = drop_original
        self.suffix = suffix
        self.medians_: Dict[str, float] = {}
        self.mads_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> "RobustScaler":
        df = _ensure_polars_df(df)
        cols = _infer_numeric(df, self.columns)
        self.feature_names_in_ = cols
        self.medians_.clear()
        self.mads_.clear()
        for c in cols:
            s = df.get_column(c).cast(pl.Float64)
            med = float(s.median())
            mad = float((s - med).abs().median())
            self.medians_[c] = med
            self.mads_[c] = mad * 1.4826 if mad > 0 else 1.0
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for c in self.feature_names_in_ or []:
            med = self.medians_[c]
            mad = self.mads_[c]
            new_c = f"{c}{self.suffix}"
            out = out.with_columns(((pl.col(c).cast(pl.Float64) - med) / mad).alias(new_c))
            if self.drop_original:
                out = out.drop(c)
        return out


class QuantileRankTransformer(Transformer):
    """Convert numeric values to their (0,1] rank within the column or per group."""

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        groupby: Optional[Sequence[str]] = None,
        method: str = "average",  # 'average' | 'min' | 'max' (ties)
        drop_original: bool = True,
        suffix: str = "__rank",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.groupby = None if groupby is None else list(groupby)
        self.method = method
        self.drop_original = drop_original
        self.suffix = suffix
        self.cols_: List[str] = []

    def fit(self, df: pl.DataFrame) -> "QuantileRankTransformer":
        df = _ensure_polars_df(df)
        self.cols_ = _infer_numeric(df, self.columns)
        for g in (self.groupby or []):
            if g not in df.columns:
                raise ValueError(f"Group column '{g}' not found")
        self.feature_names_in_ = list(set(self.cols_ + (self.groupby or [])))
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        has_groups = bool(self.groupby)
        group_keys = self.groupby if has_groups else None
        # Use rank pct via expressions
        for c in self.cols_:
            rank_expr = pl.col(c).rank(method=self.method)
            denom_expr = pl.len()
            if has_groups and group_keys is not None:
                rank_expr = rank_expr.over(group_keys)
                denom_expr = denom_expr.over(group_keys)
            expr = (rank_expr / denom_expr).alias(f"{c}{self.suffix}")
            out = out.with_columns(expr)
            if self.drop_original:
                out = out.drop(c)
        return out
