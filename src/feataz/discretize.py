from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import math
import polars as pl

from .base import Transformer, _ensure_polars_df


def _infer_numeric_columns(df: pl.DataFrame, columns: Optional[Sequence[str]]) -> List[str]:
    if columns is not None:
        return list(columns)
    nums: List[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if pl.datatypes.is_numeric(dtype):
            nums.append(name)
    return nums


def _make_bins_safe(edges: List[float]) -> List[float]:
    # ensure sorted and strictly increasing (dedupe)
    edges_sorted = sorted(set(float(e) for e in edges))
    # if not enough edges, duplicate min/max
    if len(edges_sorted) < 2:
        mn = edges_sorted[0] if edges_sorted else -float("inf")
        mx = edges_sorted[0] if edges_sorted else float("inf")
        return [mn, mx]
    return edges_sorted


class EqualFrequencyDiscretizer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        n_bins: int = 5,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__qbin",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.n_bins = int(n_bins)
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}

    def fit(self, df: pl.DataFrame) -> "EqualFrequencyDiscretizer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.bins_.clear()
        qs = [i / self.n_bins for i in range(1, self.n_bins)]
        for col in cols:
            quantiles = df.select([pl.col(col).quantile(q) for q in qs]).row(0)
            edges = [-float("inf")] + [float(q) for q in quantiles] + [float("inf")]
            self.bins_[col] = _make_bins_safe(edges)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            edges = self.bins_[col]
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(
                pl.col(col).cast(pl.Float64).cut(bins=edges, labels=labels).alias(new_col)
            )
            if self.drop_original:
                out = out.drop(col)
        return out


class EqualWidthDiscretizer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        n_bins: int = 5,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__wbin",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.n_bins = int(n_bins)
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}

    def fit(self, df: pl.DataFrame) -> "EqualWidthDiscretizer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.bins_.clear()
        for col in cols:
            s = df.get_column(col).cast(pl.Float64)
            mn = float(s.min())
            mx = float(s.max())
            if math.isfinite(mn) and math.isfinite(mx) and mx > mn:
                step = (mx - mn) / self.n_bins
                edges = [mn + i * step for i in range(self.n_bins + 1)]
            else:
                edges = [mn, mx]
            edges[0] = -float("inf")
            edges[-1] = float("inf")
            self.bins_[col] = _make_bins_safe(edges)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            edges = self.bins_[col]
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class ArbitraryDiscretizer(Transformer):
    def __init__(
        self,
        bins: Sequence[float] | Dict[str, Sequence[float]],
        columns: Optional[Sequence[str]] = None,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__abin",
    ) -> None:
        self.bins_input = bins
        self.columns = None if columns is None else list(columns)
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}

    def fit(self, df: pl.DataFrame) -> "ArbitraryDiscretizer":
        df = _ensure_polars_df(df)
        if isinstance(self.bins_input, dict):
            mapping = {k: _make_bins_safe(list(v)) for k, v in self.bins_input.items()}
            cols = list(mapping.keys()) if self.columns is None else list(self.columns)
            self.bins_ = {c: mapping[c] for c in cols}
        else:
            cols = _infer_numeric_columns(df, self.columns)
            edges = _make_bins_safe(list(self.bins_input))
            self.bins_ = {c: edges for c in cols}
        self.feature_names_in_ = list(self.bins_.keys())
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col, edges in self.bins_.items():
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class DecisionTreeDiscretizer(Transformer):
    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        problem: str = "auto",
        max_leaf_nodes: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__dtbin",
        random_state: Optional[int] = 42,
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.problem = problem
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.random_state = random_state
        self.bins_: Dict[str, List[float]] = {}

    def fit(self, df: pl.DataFrame) -> "DecisionTreeDiscretizer":
        try:
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "DecisionTreeDiscretizer requires scikit-learn. Install with `pip install scikit-learn`."
            ) from e

        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        y = df.get_column(self.target).to_numpy()
        # Determine problem if auto
        problem = self.problem
        if problem == "auto":
            n_unique = len(pl.Series(y).drop_nulls().unique())
            if n_unique <= 20 and pl.Series(y).dtype in (pl.Int64, pl.Int32, pl.UInt32, pl.UInt64, pl.Boolean):
                problem = "classification"
            else:
                problem = "regression"
        self.bins_.clear()
        for col in cols:
            X = df.get_column(col).to_numpy().reshape(-1, 1)
            if problem == "classification":
                tree = DecisionTreeClassifier(
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
            else:
                tree = DecisionTreeRegressor(
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
            tree.fit(X, y)
            thresholds = [t for t in getattr(tree.tree_, "threshold").tolist() if t > -2]
            edges = [-float("inf")] + sorted(set(float(t) for t in thresholds)) + [float("inf")]
            self.bins_[col] = _make_bins_safe(edges)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col, edges in self.bins_.items():
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class GeometricWidthDiscretizer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        n_bins: int = 5,
        drop_original: bool = True,
        labels_as_int: bool = True,
        epsilon: float = 1e-9,
        suffix: str = "__gbin",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.n_bins = int(n_bins)
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.epsilon = float(epsilon)
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}
        self.offsets_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> "GeometricWidthDiscretizer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.bins_.clear()
        self.offsets_.clear()
        for col in cols:
            s = df.get_column(col).cast(pl.Float64)
            mn = float(s.min())
            mx = float(s.max())
            offset = 0.0
            if mn <= 0:
                offset = -mn + self.epsilon
            mn_p = mn + offset
            mx_p = mx + offset
            mn_p = max(mn_p, self.epsilon)
            mx_p = max(mx_p, mn_p + self.epsilon)
            r = (mx_p / mn_p) ** (1.0 / self.n_bins)
            edges_pos = [mn_p * (r**i) for i in range(self.n_bins + 1)]
            edges = [e - offset for e in edges_pos]
            edges[0] = -float("inf")
            edges[-1] = float("inf")
            self.bins_[col] = _make_bins_safe(edges)
            self.offsets_[col] = offset
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col, edges in self.bins_.items():
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out

