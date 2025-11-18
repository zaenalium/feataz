from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import polars as pl

from .base import Transformer, _ensure_polars_df


def _infer_numeric(df: pl.DataFrame, include_bool: bool = True) -> List[str]:
    cols: List[str] = []
    for n, t in zip(df.columns, df.dtypes):
        if t.is_numeric():
            if not include_bool and t == pl.Boolean:
                continue
            cols.append(n)
    return cols


def _infer_categorical(df: pl.DataFrame) -> List[str]:
    cols: List[str] = []
    for n, t in zip(df.columns, df.dtypes):
        if t == pl.String or t == pl.Categorical:
            cols.append(n)
    return cols


class MathFeatures(Transformer):
    """Create new features from numeric columns using math operations.

    Parameters
    - columns: numeric columns to use (infers if None)
    - binary_ops: subset of {"add","sub","mul","div"}
    - unary_ops: subset of {"log","sqrt","abs"}
    - powers: list of positive integers to raise each column to (unary)
    - epsilon: small constant to stabilize division/log
    - drop_original: whether to drop source columns
    - name_sep: separator in new column names
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        binary_ops: Sequence[str] = ("add", "sub", "mul", "div"),
        unary_ops: Sequence[str] = ("log", "sqrt", "abs"),
        powers: Sequence[int] = (),
        epsilon: float = 1e-9,
        drop_original: bool = False,
        name_sep: str = "__",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.binary_ops = list(binary_ops)
        self.unary_ops = list(unary_ops)
        self.powers = [int(p) for p in powers]
        self.epsilon = float(epsilon)
        self.drop_original = drop_original
        self.name_sep = name_sep
        self.numeric_cols_: List[str] = []

    def fit(self, df: pl.DataFrame) -> "MathFeatures":
        df = _ensure_polars_df(df)
        self.numeric_cols_ = _infer_numeric(df) if self.columns is None else list(self.columns)
        for c in self.numeric_cols_:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found")
        self.feature_names_in_ = list(self.numeric_cols_)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        cols = self.numeric_cols_
        eps = self.epsilon
        out = df

        exprs: List[pl.Expr] = []

        # Unary
        for c in cols:
            base = pl.col(c).cast(pl.Float64)
            if "log" in self.unary_ops:
                exprs.append(((base + eps).log()).alias(self.name_sep.join([c, "log"])))
            if "sqrt" in self.unary_ops:
                exprs.append((base.abs().sqrt()).alias(self.name_sep.join([c, "sqrt"])))
            if "abs" in self.unary_ops:
                exprs.append(base.abs().alias(self.name_sep.join([c, "abs"])))
            for p in self.powers:
                exprs.append((base ** p).alias(self.name_sep.join([c, f"pow{p}"])))

        # Binary pairwise
        n = len(cols)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = cols[i], cols[j]
                ca = pl.col(a).cast(pl.Float64)
                cb = pl.col(b).cast(pl.Float64)
                if "add" in self.binary_ops:
                    exprs.append((ca + cb).alias(self.name_sep.join([a, "add", b])))
                if "sub" in self.binary_ops:
                    exprs.append((ca - cb).alias(self.name_sep.join([a, "sub", b])))
                if "mul" in self.binary_ops:
                    exprs.append((ca * cb).alias(self.name_sep.join([a, "mul", b])))
                if "div" in self.binary_ops:
                    exprs.append((ca / (cb + eps)).alias(self.name_sep.join([a, "div", b])))

        if exprs:
            out = out.with_columns(exprs)

        if self.drop_original:
            out = out.select([c for c in out.columns if c not in set(cols)])
        return out


class RelativeFeatures(Transformer):
    """Combine target columns with reference columns (row-wise).

    Operations: ratio (target / ref), diff (target - ref), pct_diff ((target-ref)/(|ref|+eps)).
    """

    def __init__(
        self,
        target_columns: Optional[Sequence[str]] = None,
        reference_columns: Optional[Sequence[str]] = None,
        operations: Sequence[str] = ("ratio", "diff", "pct_diff"),
        epsilon: float = 1e-9,
        drop_original: bool = False,
        name_sep: str = "__",
        ref_token: str = "ref",
    ) -> None:
        self.target_columns = None if target_columns is None else list(target_columns)
        self.reference_columns = None if reference_columns is None else list(reference_columns)
        self.operations = list(operations)
        self.epsilon = float(epsilon)
        self.drop_original = drop_original
        self.name_sep = name_sep
        self.ref_token = ref_token
        self.targets_: List[str] = []
        self.refs_: List[str] = []

    def fit(self, df: pl.DataFrame) -> "RelativeFeatures":
        df = _ensure_polars_df(df)
        # infer numeric for missing sets
        num_cols = _infer_numeric(df)
        self.targets_ = num_cols if self.target_columns is None else list(self.target_columns)
        self.refs_ = num_cols if self.reference_columns is None else list(self.reference_columns)
        for c in self.targets_ + self.refs_:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found")
        self.feature_names_in_ = list(set(self.targets_ + self.refs_))
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        eps = self.epsilon
        out = df
        exprs: List[pl.Expr] = []
        for t in self.targets_:
            ct = pl.col(t).cast(pl.Float64)
            for r in self.refs_:
                cr = pl.col(r).cast(pl.Float64)
                base = [t, r]
                if "ratio" in self.operations:
                    name = self.name_sep.join([t, "ratio", self.ref_token, r])
                    exprs.append((ct / (cr + eps)).alias(name))
                if "diff" in self.operations:
                    name = self.name_sep.join([t, "diff", self.ref_token, r])
                    exprs.append((ct - cr).alias(name))
                if "pct_diff" in self.operations:
                    name = self.name_sep.join([t, "pctdiff", self.ref_token, r])
                    exprs.append(((ct - cr) / (cr.abs() + eps)).alias(name))
        if exprs:
            out = out.with_columns(exprs)
        if self.drop_original:
            out = out.select([c for c in out.columns if c not in set(self.feature_names_in_ or [])])
        return out


class CyclicalFeatures(Transformer):
    """Encode numeric cyclical variables into sin/cos features.

    Parameters
    - columns: columns to transform (numeric). If None, no inference is applied; must be provided.
    - period: float applied to all columns if `periods` not provided.
    - periods: dict mapping column -> period.
    - offset: optional phase shift added before scaling.
    - drop_original: drop the source columns after adding sin/cos.
    """

    def __init__(
        self,
        columns: Sequence[str],
        period: Optional[float] = None,
        periods: Optional[Dict[str, float]] = None,
        offset: float = 0.0,
        drop_original: bool = False,
        name_sep: str = "__",
    ) -> None:
        self.columns = list(columns)
        self.period = period
        self.periods = dict(periods) if periods is not None else None
        self.offset = float(offset)
        self.drop_original = drop_original
        self.name_sep = name_sep
        self.periods_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> "CyclicalFeatures":
        df = _ensure_polars_df(df)
        for c in self.columns:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found")
        if self.periods is not None:
            self.periods_ = {k: float(v) for k, v in self.periods.items()}
        elif self.period is not None:
            self.periods_ = {c: float(self.period) for c in self.columns}
        else:
            raise ValueError("Provide either 'period' or 'periods'")
        self.feature_names_in_ = list(self.columns)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for c in self.columns:
            period = self.periods_[c]
            angle = (pl.col(c).cast(pl.Float64) + self.offset) * (2.0 * math.pi / period)
            out = out.with_columns([
                angle.sin().alias(self.name_sep.join([c, "sin"])),
                angle.cos().alias(self.name_sep.join([c, "cos"])),
            ])
            if self.drop_original:
                out = out.drop(c)
        return out


class DecisionTreeFeatures(Transformer):
    """Create features from Decision Tree predictions over one or more feature sets.

    Trains one tree per feature set and appends predictions (or probabilities) as new columns.
    """

    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        feature_sets: Optional[Sequence[Sequence[str]]] = None,
        problem: str = "auto",  # 'classification' | 'regression' | 'auto'
        output: str = "auto",  # 'predict' | 'proba' | 'auto'
        class_index: int = 1,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        drop_original: bool = False,
        name_sep: str = "__",
        set_sep: str = "__and__",
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.feature_sets = [list(s) for s in feature_sets] if feature_sets is not None else None
        self.problem = problem
        self.output = output
        self.class_index = int(class_index)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.drop_original = drop_original
        self.name_sep = name_sep
        self.set_sep = set_sep

        self.trees_: List[Tuple[object, List[str], str, str]] = []
        self.ordinal_maps_: Dict[str, Dict[str, int]] = {}

    def _make_feature_sets(self, df: pl.DataFrame) -> List[List[str]]:
        if self.feature_sets is not None:
            return [list(s) for s in self.feature_sets]
        cols = self.columns if self.columns is not None else _infer_numeric(df)
        return [list(cols)] if cols else []

    def _encode_column(self, s: pl.Series, name: str) -> pl.Series:
        # Ordinal-encode string-like columns for sklearn
        if s.dtype == pl.String or s.dtype == pl.Categorical:
            mapping = self.ordinal_maps_.get(name)
            if mapping is None:
                cats = s.drop_nulls().cast(pl.Utf8).unique().to_list()
                mapping = {c: i for i, c in enumerate(sorted(cats))}
                self.ordinal_maps_[name] = mapping
            return s.cast(pl.Utf8).map_elements(lambda x, m=mapping: m.get(x, -1), return_dtype=pl.Int64)
        return s.cast(pl.Float64)

    def fit(self, df: pl.DataFrame) -> "DecisionTreeFeatures":
        try:
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "DecisionTreeFeatures requires scikit-learn. Install with `pip install scikit-learn`."
            ) from e

        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        y = df.get_column(self.target).to_numpy()

        # Decide problem
        problem = self.problem
        if problem == "auto":
            n_unique = len(pl.Series(y).drop_nulls().unique())
            if n_unique <= 20 and pl.Series(y).dtype in (pl.Int64, pl.Int32, pl.UInt32, pl.UInt64, pl.Boolean):
                problem = "classification"
            else:
                problem = "regression"

        feature_sets = self._make_feature_sets(df)
        if not feature_sets:
            raise ValueError("No feature columns provided/inferred for DecisionTreeFeatures")

        self.trees_.clear()
        self.ordinal_maps_.clear()
        for fs in feature_sets:
            # Build encoded matrix
            enc_df = pl.DataFrame({c: self._encode_column(df.get_column(c), c) for c in fs})
            X = enc_df.to_numpy()
            if problem == "classification":
                tree = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
                tree.fit(X, y)
            else:
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
                tree.fit(X, y)
            self.trees_.append((tree, fs, problem, self.output))

        self.feature_names_in_ = list({c for _, fs, _, _ in self.trees_ for c in fs} | {self.target})
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for tree, fs, problem, output in self.trees_:
            enc_df = pl.DataFrame({c: self._encode_column(df.get_column(c), c) for c in fs})
            X = enc_df.to_numpy()
            out_name_base = self.set_sep.join(fs)
            # Decide output
            use_output = output
            if use_output == "auto":
                use_output = "proba" if problem == "classification" else "predict"
            try:
                if problem == "classification" and use_output == "proba":
                    import numpy as np
                    proba = tree.predict_proba(X)
                    idx = min(self.class_index, proba.shape[1] - 1)
                    yhat = proba[:, idx]
                    name = self.name_sep.join([out_name_base, "tree", "proba"])
                else:
                    yhat = tree.predict(X)
                    name = self.name_sep.join([out_name_base, "tree", "pred"])
            except Exception:
                yhat = tree.predict(X)
                name = self.name_sep.join([out_name_base, "tree", "pred"])
            out = out.with_columns(pl.Series(name=name, values=yhat))
        if self.drop_original:
            keep = [c for c in out.columns if c not in set(self.feature_names_in_ or [])]
            out = out.select(keep)
        return out

