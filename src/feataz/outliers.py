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


class ClipOutliers(Transformer):
    """Clip/flag/remove outliers with common statistics.

    Methods:
    - 'quantile': clip to [q_low, q_high]
    - 'iqr': clip to [Q1 - k*IQR, Q3 + k*IQR]
    - 'zscore': clip to [mean - k*std, mean + k*std]
    - 'mad': clip to [median - k*1.4826*MAD, median + k*1.4826*MAD]
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        method: str = "quantile",
        q_low: float = 0.01,
        q_high: float = 0.99,
        iqr_factor: float = 1.5,
        z_thresh: float = 3.0,
        mad_thresh: float = 3.5,
        action: str = "clip",  # 'clip' | 'flag' | 'remove'
        flag_suffix: str = "__outlier",
        config: Optional[Dict[str, Dict[str, float | str]]] = None,
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.method = method
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.iqr_factor = float(iqr_factor)
        self.z_thresh = float(z_thresh)
        self.mad_thresh = float(mad_thresh)
        self.action = action
        self.flag_suffix = flag_suffix
        self.config = config or {}
        self.bounds_: Dict[str, tuple[float, float]] = {}

    def fit(self, df: pl.DataFrame) -> "ClipOutliers":
        df = _ensure_polars_df(df)
        cols = _infer_numeric(df, self.columns)
        self.feature_names_in_ = cols
        self.bounds_.clear()
        for c in cols:
            s = df.get_column(c).cast(pl.Float64)
            method = str(self.config.get(c, {}).get("method", self.method))
            if method == "quantile":
                low = float(s.quantile(self.q_low))
                high = float(s.quantile(self.q_high))
            elif method == "iqr":
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                fac = float(self.config.get(c, {}).get("iqr_factor", self.iqr_factor))
                low = q1 - fac * iqr
                high = q3 + fac * iqr
            elif method == "zscore":
                mu = float(s.mean())
                sd = float(s.std())
                zt = float(self.config.get(c, {}).get("z_thresh", self.z_thresh))
                low = mu - zt * sd
                high = mu + zt * sd
            elif method == "mad":
                med = float(s.median())
                mad = float((s - med).abs().median())
                sigma = 1.4826 * mad
                mt = float(self.config.get(c, {}).get("mad_thresh", self.mad_thresh))
                low = med - mt * sigma
                high = med + mt * sigma
            else:
                raise ValueError("Unsupported method")
            self.bounds_[c] = (low, high)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        if self.action == "clip":
            exprs: List[pl.Expr] = []
            for c, (low, high) in self.bounds_.items():
                exprs.append(pl.col(c).clip(low, high).alias(c))
            if exprs:
                out = out.with_columns(exprs)
            return out
        elif self.action == "flag":
            exprs: List[pl.Expr] = []
            for c, (low, high) in self.bounds_.items():
                flag = ((pl.col(c) < low) | (pl.col(c) > high)).alias(f"{c}{self.flag_suffix}")
                exprs.append(flag)
            if exprs:
                out = out.with_columns(exprs)
            return out
        elif self.action == "remove":
            masks: List[pl.Expr] = []
            for c, (low, high) in self.bounds_.items():
                masks.append((pl.col(c) < low) | (pl.col(c) > high))
            if masks:
                # remove rows that are outliers in any selected column
                combined = masks[0]
                for m in masks[1:]:
                    combined = combined | m
                out = out.filter(~combined)
            return out
        else:
            raise ValueError("Unsupported action")


class IsolationForestOutlierHandler(Transformer):
    """Detect outliers via IsolationForest (requires scikit-learn).

    - action: 'flag' adds a boolean column; 'remove' drops outlier rows
    - add_score: adds raw score from decision_function (higher = less abnormal)
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        contamination: float = 0.01,
        random_state: Optional[int] = 42,
        action: str = "flag",
        add_score: bool = False,
        flag_col: str = "iso_outlier",
        score_col: str = "iso_score",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.contamination = float(contamination)
        self.random_state = random_state
        self.action = action
        self.add_score = add_score
        self.flag_col = flag_col
        self.score_col = score_col
        self.cols_: List[str] = []
        self.model_: object | None = None
        self.medians_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> "IsolationForestOutlierHandler":
        try:
            from sklearn.ensemble import IsolationForest
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("IsolationForestOutlierHandler requires scikit-learn") from e

        df = _ensure_polars_df(df)
        self.cols_ = _infer_numeric(df, self.columns)
        # median-impute for fitting
        Xdf = df.select(self.cols_).cast(pl.Float64)
        for c in self.cols_:
            med = float(Xdf.get_column(c).median())
            self.medians_[c] = med
        # median-impute nulls per column for model fitting
        X = Xdf.with_columns([pl.col(c).fill_null(self.medians_[c]) for c in self.cols_]).to_numpy()
        model = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        model.fit(X)
        self.model_ = model
        self.feature_names_in_ = list(self.cols_)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError("Call fit before transform")
        out = df
        Xdf = df.select(self.cols_).cast(pl.Float64)
        X = Xdf.with_columns([pl.col(c).fill_null(self.medians_[c]) for c in self.cols_]).to_numpy()
        import numpy as np

        labels = self.model_.predict(X)  # type: ignore[attr-defined]
        is_out = labels == -1
        if self.action == "remove":
            mask = pl.Series(name=self.flag_col, values=~is_out)
            out = out.filter(mask)
            if self.add_score:
                scores = self.model_.decision_function(X)  # type: ignore[attr-defined]
                out = out.with_columns(pl.Series(name=self.score_col, values=scores))
            return out
        else:  # flag
            out = out.with_columns(pl.Series(name=self.flag_col, values=is_out))
            if self.add_score:
                scores = self.model_.decision_function(X)  # type: ignore[attr-defined]
                out = out.with_columns(pl.Series(name=self.score_col, values=scores))
            return out
