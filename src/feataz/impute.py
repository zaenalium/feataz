from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import polars as pl

from .base import Transformer, _ensure_polars_df


def _infer_numeric(df: pl.DataFrame, exclude: Sequence[str] = ()) -> List[str]:
    cols: List[str] = []
    ex = set(exclude)
    for n, t in zip(df.columns, df.dtypes):
        if n in ex:
            continue
        if pl.datatypes.is_numeric(t):
            cols.append(n)
    return cols


def _infer_categorical(df: pl.DataFrame, exclude: Sequence[str] = ()) -> List[str]:
    cols: List[str] = []
    ex = set(exclude)
    for n, t in zip(df.columns, df.dtypes):
        if n in ex:
            continue
        if pl.datatypes.is_string_dtype(t) or t == pl.Categorical:
            cols.append(n)
    return cols


class SimpleImputer(Transformer):
    """Simple missing value imputation per column.

    - Numeric: strategy in {mean, median, constant}
    - Categorical: strategy in {most_frequent, constant}
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        numeric_strategy_map: Optional[Dict[str, str]] = None,
        categorical_strategy_map: Optional[Dict[str, str]] = None,
        fill_value_num: float | int | None = None,
        fill_value_cat: str | None = None,
        fill_value_map: Optional[Dict[str, Any]] = None,
        add_indicator: bool = False,
        indicator_suffix: str = "__isnull",
        drop_original: bool = False,
        suffix: str = "__imp",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_strategy_map = numeric_strategy_map or {}
        self.categorical_strategy_map = categorical_strategy_map or {}
        self.fill_value_num = fill_value_num
        self.fill_value_cat = fill_value_cat
        self.fill_value_map = fill_value_map or {}
        self.add_indicator = add_indicator
        self.indicator_suffix = indicator_suffix
        self.drop_original = drop_original
        self.suffix = suffix
        self.values_: Dict[str, object] = {}

    def fit(self, df: pl.DataFrame) -> "SimpleImputer":
        df = _ensure_polars_df(df)
        cols = list(self.columns) if self.columns is not None else list(df.columns)
        self.feature_names_in_ = cols
        self.values_.clear()
        for c in cols:
            dt = df.get_column(c).dtype
            s = df.get_column(c)
            if pl.datatypes.is_numeric(dt):
                strategy = self.numeric_strategy_map.get(c, self.numerical_strategy)
                if strategy == "mean":
                    self.values_[c] = float(s.mean())
                elif strategy == "median":
                    self.values_[c] = float(s.median())
                elif strategy == "constant":
                    val = self.fill_value_map.get(c, self.fill_value_num)
                    if val is None:
                        raise ValueError("fill_value_num must be set for constant strategy")
                    self.values_[c] = val
                else:
                    raise ValueError("Unsupported numerical_strategy")
            elif pl.datatypes.is_string_dtype(dt) or dt == pl.Categorical:
                strategy = self.categorical_strategy_map.get(c, self.categorical_strategy)
                if strategy == "most_frequent":
                    mode_vals = s.drop_nulls().mode().to_list()
                    mode = mode_vals[0] if mode_vals else "__missing__"
                    self.values_[c] = mode
                elif strategy == "constant":
                    val = self.fill_value_map.get(c, self.fill_value_cat)
                    if val is None:
                        raise ValueError("fill_value_cat must be set for constant strategy")
                    self.values_[c] = val
                else:
                    raise ValueError("Unsupported categorical_strategy")
            else:
                # unsupported dtype: leave as-is
                continue
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for c, val in self.values_.items():
            if self.add_indicator and c in df.columns:
                out = out.with_columns(pl.col(c).is_null().alias(f"{c}{self.indicator_suffix}"))
            new_c = c if self.drop_original else f"{c}{self.suffix}"
            out = out.with_columns(pl.col(c).fill_null(val).alias(new_c))
            if self.drop_original and new_c != c:
                out = out.drop(c)
        return out


class GroupImputer(Transformer):
    """Impute missing values using group-level statistics, with global fallback.

    - groupby: list of columns to group by
    - strategy: numeric -> {mean, median}; categorical -> {most_frequent}
    - fallback: global SimpleImputer strategy to use if group stat is null
    """

    def __init__(
        self,
        groupby: Sequence[str],
        columns: Optional[Sequence[str]] = None,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        numeric_strategy_map: Optional[Dict[str, str]] = None,
        categorical_strategy_map: Optional[Dict[str, str]] = None,
        add_indicator: bool = False,
        indicator_suffix: str = "__isnull",
        drop_original: bool = True,
        suffix: str = "__grpimp",
    ) -> None:
        self.groupby = list(groupby)
        self.columns = None if columns is None else list(columns)
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_strategy_map = numeric_strategy_map or {}
        self.categorical_strategy_map = categorical_strategy_map or {}
        self.add_indicator = add_indicator
        self.indicator_suffix = indicator_suffix
        self.drop_original = drop_original
        self.suffix = suffix
        self.group_stats_: pl.DataFrame | None = None
        self.fallback_values_: Dict[str, object] = {}

    def fit(self, df: pl.DataFrame) -> "GroupImputer":
        df = _ensure_polars_df(df)
        for g in self.groupby:
            if g not in df.columns:
                raise ValueError(f"Group column '{g}' not found")
        cols = list(self.columns) if self.columns is not None else [c for c in df.columns if c not in self.groupby]
        self.feature_names_in_ = list(set(cols + self.groupby))

        # Compute group stats
        exprs: List[pl.Expr] = []
        for c in cols:
            dt = df.get_column(c).dtype
            if pl.datatypes.is_numeric(dt):
                strategy = self.numeric_strategy_map.get(c, self.numeric_strategy)
                if strategy == "mean":
                    exprs.append(pl.col(c).mean().alias(f"{c}{self.suffix}_val"))
                elif strategy == "median":
                    exprs.append(pl.col(c).median().alias(f"{c}{self.suffix}_val"))
                else:
                    raise ValueError("Unsupported numeric_strategy")
            elif pl.datatypes.is_string_dtype(dt) or dt == pl.Categorical:
                strategy = self.categorical_strategy_map.get(c, self.categorical_strategy)
                if strategy == "most_frequent":
                    exprs.append(pl.col(c).mode().arr.first().alias(f"{c}{self.suffix}_val"))
                else:
                    raise ValueError("Unsupported categorical_strategy for categorical columns")
        grp = df.group_by(self.groupby).agg(exprs)
        self.group_stats_ = grp

        # Build global fallback values
        self.fallback_values_.clear()
        simple = SimpleImputer(
            columns=cols,
            numerical_strategy=self.numeric_strategy if self.numeric_strategy in {"mean", "median"} else "median",
            categorical_strategy=self.categorical_strategy if self.categorical_strategy in {"most_frequent", "constant"} else "most_frequent",
        ).fit(df)
        self.fallback_values_ = dict(simple.values_)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_ or self.group_stats_ is None:
            raise RuntimeError("Call fit before transform")
        out = df.join(self.group_stats_, on=self.groupby, how="left")
        # fill values
        for c in [col for col in df.columns if col not in self.groupby]:
            fill_col = f"{c}{self.suffix}_val"
            if fill_col in out.columns:
                if self.add_indicator and c in df.columns:
                    out = out.with_columns(pl.col(c).is_null().alias(f"{c}{self.indicator_suffix}"))
                new_c = c if self.drop_original else f"{c}{self.suffix}"
                out = out.with_columns(
                    pl.when(pl.col(c).is_null())
                    .then(pl.coalesce([pl.col(fill_col), pl.lit(self.fallback_values_.get(c))]))
                    .otherwise(pl.col(c))
                    .alias(new_c)
                )
                if self.drop_original and new_c != c:
                    out = out.drop(c)
        # drop temp fill columns
        drop_cols = [c for c in out.columns if c.endswith(f"{self.suffix}_val")]
        if drop_cols:
            out = out.drop(drop_cols)
        return out


class KNNImputer(Transformer):
    """KNN-based imputation for numeric columns (requires scikit-learn)."""

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        n_neighbors: int = 5,
        weights: str = "uniform",
        drop_original: bool = True,
        suffix: str = "__knnimp",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.drop_original = drop_original
        self.suffix = suffix
        self.cols_: List[str] = []
        self.imputer_: object | None = None

    def fit(self, df: pl.DataFrame) -> "KNNImputer":
        try:
            from sklearn.impute import KNNImputer as _KNN
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("KNNImputer requires scikit-learn") from e

        df = _ensure_polars_df(df)
        self.cols_ = _infer_numeric(df) if self.columns is None else list(self.columns)
        X = df.select(self.cols_).to_numpy()
        imp = _KNN(n_neighbors=self.n_neighbors, weights=self.weights)
        imp.fit(X)
        self.imputer_ = imp
        self.feature_names_in_ = list(self.cols_)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_ or self.imputer_ is None:
            raise RuntimeError("Call fit before transform")
        out = df
        X = df.select(self.cols_).to_numpy()
        import numpy as np

        X_imp = self.imputer_.transform(X)  # type: ignore[attr-defined]
        for i, c in enumerate(self.cols_):
            series = pl.Series(name=c if self.drop_original else f"{c}{self.suffix}", values=X_imp[:, i])
            out = out.with_columns(series)
            if self.drop_original and series.name != c:
                out = out.drop(c)
        return out


class IterativeImputer(Transformer):
    """Multivariate imputer (MICE) for numeric columns (requires scikit-learn)."""

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        max_iter: int = 10,
        random_state: Optional[int] = 0,
        drop_original: bool = True,
        suffix: str = "__mice",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.drop_original = drop_original
        self.suffix = suffix
        self.cols_: List[str] = []
        self.imputer_: object | None = None

    def fit(self, df: pl.DataFrame) -> "IterativeImputer":
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer as _MICE
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError("IterativeImputer requires scikit-learn >= 0.21") from e

        df = _ensure_polars_df(df)
        self.cols_ = _infer_numeric(df) if self.columns is None else list(self.columns)
        X = df.select(self.cols_).to_numpy()
        imp = _MICE(max_iter=self.max_iter, random_state=self.random_state)
        imp.fit(X)
        self.imputer_ = imp
        self.feature_names_in_ = list(self.cols_)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_ or self.imputer_ is None:
            raise RuntimeError("Call fit before transform")
        out = df
        X = df.select(self.cols_).to_numpy()
        X_imp = self.imputer_.transform(X)  # type: ignore[attr-defined]
        for i, c in enumerate(self.cols_):
            series = pl.Series(name=c if self.drop_original else f"{c}{self.suffix}", values=X_imp[:, i])
            out = out.with_columns(series)
            if self.drop_original and series.name != c:
                out = out.drop(c)
        return out


class TimeSeriesImputer(Transformer):
    """Fill missing values via forward/backward fill over time, optionally per group.

    - method: 'ffill' | 'bfill' | 'both'
    - limit is not supported explicitly in Polars; fills span entire gaps.
    """

    def __init__(
        self,
        time_column: str,
        columns: Optional[Sequence[str]] = None,
        groupby: Optional[Sequence[str]] = None,
        method: str = "both",
        drop_original: bool = True,
        suffix: str = "__tsimp",
    ) -> None:
        self.time_column = time_column
        self.columns = None if columns is None else list(columns)
        self.groupby = None if groupby is None else list(groupby)
        self.method = method
        self.drop_original = drop_original
        self.suffix = suffix
        self.cols_: List[str] = []

    def fit(self, df: pl.DataFrame) -> "TimeSeriesImputer":
        df = _ensure_polars_df(df)
        if self.time_column not in df.columns:
            raise ValueError(f"time column '{self.time_column}' not found")
        self.cols_ = (
            [c for c in df.columns if c not in ([self.time_column] + (self.groupby or []))]
            if self.columns is None
            else list(self.columns)
        )
        self.feature_names_in_ = list(set(self.cols_ + [self.time_column] + (self.groupby or [])))
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        # preserve row order
        out = df.with_row_count("__row__")
        sort_keys = (self.groupby or []) + [self.time_column]
        work = out.sort(sort_keys)
        exprs: List[pl.Expr] = []
        over = self.groupby
        for c in self.cols_:
            e = pl.col(c)
            if self.method in ("ffill", "both"):
                e = e.fill_null(strategy="forward")
            if self.method in ("bfill", "both"):
                e = e.fill_null(strategy="backward")
            if over:
                e = e.over(over)
            new_c = c if self.drop_original else f"{c}{self.suffix}"
            exprs.append(e.alias(new_c))
        work = work.with_columns(exprs)
        work = work.sort("__row__").drop("__row__")
        if self.drop_original:
            return work
        else:
            # keep originals as well
            return work
