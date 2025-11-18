from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import math
import numpy as np
import polars as pl

from .base import Transformer, _ensure_polars_df
from .discretize import _infer_numeric_columns


class LogTransformer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        epsilon: float = 1e-9,
        drop_original: bool = True,
        suffix: str = "__log",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.epsilon = float(epsilon)
        self.drop_original = drop_original
        self.suffix = suffix
        self.offsets_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> "LogTransformer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.offsets_.clear()
        for col in cols:
            mn = float(df.get_column(col).min())
            offset = 0.0
            if mn <= 0:
                offset = -mn + self.epsilon
            self.offsets_[col] = offset
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            offset = self.offsets_[col]
            new_col = f"{col}{self.suffix}"
            out = out.with_columns((pl.col(col).cast(pl.Float64) + offset).log().alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class LogCPTransformer(Transformer):
    """Log(x + c) with automatic offset to keep positivity."""

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        c: float = 1.0,
        epsilon: float = 1e-9,
        drop_original: bool = True,
        suffix: str = "__logcp",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.c = float(c)
        self.epsilon = float(epsilon)
        self.drop_original = drop_original
        self.suffix = suffix
        self.offsets_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> "LogCPTransformer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.offsets_.clear()
        for col in cols:
            mn = float(df.get_column(col).min())
            offset = 0.0
            if mn + self.c <= 0:
                offset = -(mn + self.c) + self.epsilon
            self.offsets_[col] = offset
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            offset = self.offsets_[col]
            new_col = f"{col}{self.suffix}"
            out = out.with_columns((pl.col(col).cast(pl.Float64) + self.c + offset).log().alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class ReciprocalTransformer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        epsilon: float = 1e-9,
        drop_original: bool = True,
        suffix: str = "__rec",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.epsilon = float(epsilon)
        self.drop_original = drop_original
        self.suffix = suffix
        self.offsets_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> "ReciprocalTransformer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.offsets_.clear()
        for col in cols:
            has_zero = df.get_column(col).cast(pl.Float64).is_in([0.0]).sum() > 0
            offset = self.epsilon if bool(has_zero) else 0.0
            self.offsets_[col] = offset
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            offset = self.offsets_[col]
            new_col = f"{col}{self.suffix}"
            out = out.with_columns((1.0 / (pl.col(col).cast(pl.Float64) + offset)).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class ArcsinTransformer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        scale: bool = False,
        drop_original: bool = True,
        suffix: str = "__arcsin",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.scale = bool(scale)
        self.drop_original = drop_original
        self.suffix = suffix
        self.mins_: Dict[str, float] = {}
        self.maxs_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> "ArcsinTransformer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.mins_.clear()
        self.maxs_.clear()
        for col in cols:
            s = df.get_column(col).cast(pl.Float64)
            mn = float(s.min())
            mx = float(s.max())
            if not self.scale:
                if mn < 0.0 or mx > 1.0:
                    raise ValueError(
                        f"Column {col} must be within [0,1] or set scale=True"
                    )
            self.mins_[col] = mn
            self.maxs_[col] = mx
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            if self.scale:
                mn = self.mins_[col]
                mx = self.maxs_[col]
                denom = (mx - mn) if mx > mn else 1.0
                expr = ((pl.col(col).cast(pl.Float64) - mn) / denom).sqrt().arcsin()
            else:
                expr = pl.col(col).cast(pl.Float64).sqrt().arcsin()
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(expr.alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class PowerTransformer(Transformer):
    def __init__(
        self,
        power: float,
        columns: Optional[Sequence[str]] = None,
        ensure_positive: bool = True,
        epsilon: float = 1e-9,
        drop_original: bool = True,
        suffix: str = "__pow",
    ) -> None:
        self.power = float(power)
        self.columns = None if columns is None else list(columns)
        self.ensure_positive = ensure_positive
        self.epsilon = float(epsilon)
        self.drop_original = drop_original
        self.suffix = suffix
        self.offsets_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> "PowerTransformer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.offsets_.clear()
        for col in cols:
            mn = float(df.get_column(col).min())
            offset = 0.0
            if self.ensure_positive and (mn <= 0.0):
                offset = -mn + self.epsilon
            self.offsets_[col] = offset
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        p = self.power
        out = df
        for col in self.feature_names_in_ or []:
            offset = self.offsets_[col]
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(((pl.col(col).cast(pl.Float64) + offset) ** p).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


def _boxcox_transform(x: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0:
        return np.log(x)
    return (np.power(x, lam) - 1.0) / lam


def _yeojohnson_transform(x: np.ndarray, lam: float) -> np.ndarray:
    y = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    if lam == 0:
        y[pos] = np.log1p(x[pos])
    else:
        y[pos] = ((np.power(x[pos] + 1.0, lam) - 1.0) / lam)
    if lam == 2:
        y[neg] = -np.log1p(-x[neg])
    else:
        y[neg] = -((np.power(1.0 - x[neg], 2 - lam) - 1.0) / (2 - lam))
    return y


class BoxCoxTransformer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        lmbda: Optional[float] = None,
        epsilon: float = 1e-9,
        drop_original: bool = True,
        suffix: str = "__boxcox",
        grid_min: float = -5.0,
        grid_max: float = 5.0,
        grid_size: int = 101,
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.lmbda = lmbda
        self.epsilon = float(epsilon)
        self.drop_original = drop_original
        self.suffix = suffix
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.grid_size = int(grid_size)
        self.lambdas_: Dict[str, float] = {}
        self.offsets_: Dict[str, float] = {}

    def _fit_lambda(self, x: np.ndarray) -> float:
        # maximize log-likelihood across grid
        # llf = -n/2 * log(var(y)) + (lam-1) * sum(log(x))
        if self.lmbda is not None:
            return float(self.lmbda)
        lam_grid = np.linspace(self.grid_min, self.grid_max, self.grid_size)
        logx = np.log(x)
        n = x.size
        best_ll = -np.inf
        best_lam = 0.0
        for lam in lam_grid:
            y = _boxcox_transform(x, lam)
            s2 = np.var(y, ddof=1) if n > 1 else np.var(y)
            if s2 <= 0:
                continue
            ll = -(n / 2.0) * np.log(s2) + (lam - 1.0) * np.sum(logx)
            if ll > best_ll:
                best_ll = ll
                best_lam = float(lam)
        return best_lam

    def fit(self, df: pl.DataFrame) -> "BoxCoxTransformer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.lambdas_.clear()
        self.offsets_.clear()
        for col in cols:
            s = df.get_column(col).cast(pl.Float64).to_numpy()
            mn = float(np.nanmin(s))
            offset = 0.0
            if mn <= 0.0:
                offset = -mn + self.epsilon
            x = s + offset
            lam = self._fit_lambda(x)
            self.lambdas_[col] = lam
            self.offsets_[col] = offset
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            lam = self.lambdas_[col]
            offset = self.offsets_[col]
            new_col = f"{col}{self.suffix}"
            # Scalar implementation to reduce NumPy overhead per element
            def _boxcox_scalar(v: float, lam=lam, off=offset) -> float:
                x = float(v) + float(off)
                if lam == 0:
                    return float(np.log(x))
                return float((np.power(x, lam) - 1.0) / lam)

            out = out.with_columns(
                pl.col(col).cast(pl.Float64).map_elements(_boxcox_scalar, return_dtype=pl.Float64).alias(new_col)
            )
            if self.drop_original:
                out = out.drop(col)
        return out

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before inverse_transform")
        out = df
        for col in self.feature_names_in_ or []:
            lam = self.lambdas_[col]
            offset = self.offsets_[col]
            new_col = col  # inverse to original name
            def _inv_boxcox(v: float, lam=lam, off=offset) -> float:
                if lam == 0:
                    return float(np.exp(v) - off)
                return float(np.power(lam * v + 1.0, 1.0 / lam) - off)
            # assuming df contains transformed column name f"{col}{self.suffix}"
            tname = f"{col}{self.suffix}"
            if tname not in out.columns:
                continue
            out = out.with_columns(pl.col(tname).map_elements(_inv_boxcox, return_dtype=pl.Float64).alias(new_col))
        return out


class YeoJohnsonTransformer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        lmbda: Optional[float] = None,
        drop_original: bool = True,
        suffix: str = "__yj",
        grid_min: float = -5.0,
        grid_max: float = 5.0,
        grid_size: int = 101,
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.lmbda = lmbda
        self.drop_original = drop_original
        self.suffix = suffix
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.grid_size = int(grid_size)
        self.lambdas_: Dict[str, float] = {}

    def _fit_lambda(self, x: np.ndarray) -> float:
        if self.lmbda is not None:
            return float(self.lmbda)
        lam_grid = np.linspace(self.grid_min, self.grid_max, self.grid_size)
        n = x.size
        best_ll = -np.inf
        best_lam = 0.0
        for lam in lam_grid:
            y = _yeojohnson_transform(x, lam)
            s2 = np.var(y, ddof=1) if n > 1 else np.var(y)
            if s2 <= 0:
                continue
            # log-likelihood up to constant proportional to -n/2*log(var)
            ll = -(n / 2.0) * np.log(s2)
            if ll > best_ll:
                best_ll = ll
                best_lam = float(lam)
        return best_lam

    def fit(self, df: pl.DataFrame) -> "YeoJohnsonTransformer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.lambdas_.clear()
        for col in cols:
            s = df.get_column(col).cast(pl.Float64).to_numpy()
            lam = self._fit_lambda(s)
            self.lambdas_[col] = lam
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            lam = self.lambdas_[col]
            new_col = f"{col}{self.suffix}"
            def _yj_scalar(v: float, lam=lam) -> float:
                x = float(v)
                if x >= 0:
                    if lam == 0:
                        return float(np.log1p(x))
                    return float((np.power(x + 1.0, lam) - 1.0) / lam)
                else:
                    if lam == 2:
                        return float(-np.log1p(-x))
                    return float(-((np.power(1.0 - x, 2 - lam) - 1.0) / (2 - lam)))

            out = out.with_columns(
                pl.col(col).cast(pl.Float64).map_elements(_yj_scalar, return_dtype=pl.Float64).alias(new_col)
            )
            if self.drop_original:
                out = out.drop(col)
        return out

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before inverse_transform")
        out = df
        for col in self.feature_names_in_ or []:
            lam = self.lambdas_[col]
            def _inv_yj(v: float, lam=lam) -> float:
                if v is None:
                    return np.nan
                if lam == 0 and v >= 0:
                    return float(np.expm1(v))
                if lam == 2 and v < 0:
                    return float(1 - np.expm1(-v))
                if v >= 0:
                    return float(np.power(lam * v + 1.0, 1.0 / lam) - 1.0)
                else:
                    return float(1.0 - np.power(1.0 - (2 - lam) * v, 1.0 / (2 - lam)))
            tname = f"{col}{self.suffix}"
            if tname not in out.columns:
                continue
            out = out.with_columns(pl.col(tname).map_elements(_inv_yj, return_dtype=pl.Float64).alias(col))
        return out
