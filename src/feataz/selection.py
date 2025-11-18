from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from .base import Transformer, _ensure_polars_df


def _resolve_numeric_columns(df: pl.DataFrame, columns: Optional[Sequence[str]]) -> List[str]:
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Columns not found in DataFrame: {missing_str}")
        return list(columns)

    resolved: List[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if dtype.is_numeric() or dtype == pl.Boolean:
            resolved.append(name)
    return resolved


def _infer_problem_type(series: pl.Series, problem: str) -> str:
    if problem in {"classification", "regression"}:
        return problem

    dtype = series.dtype
    if dtype in {pl.Boolean, pl.Utf8, pl.Categorical, pl.String}:
        return "classification"
    if dtype.is_integer():
        try:
            unique = int(series.drop_nulls().n_unique())
        except Exception:
            unique = 0
        if unique and unique <= 10:
            return "classification"
        return "regression"
    if dtype.is_integer():
        return "regression"
    return "regression"


def _prepare_training_frame(df: pl.DataFrame, columns: Sequence[str], target: str) -> pl.DataFrame:
    return (
        df.select([*columns, target])
        .drop_nulls()
        .with_columns([pl.col(c).cast(pl.Float64) for c in columns])
    )


def _select_features(
    scores: Dict[str, float],
    ordered_features: Sequence[str],
    k: Optional[int],
    threshold: Optional[float],
) -> Tuple[List[str], List[str]]:
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected: List[str] = []
    for name, score in sorted_items:
        if threshold is not None and score <= threshold:
            continue
        selected.append(name)
    if k is not None:
        selected = selected[: max(0, min(k, len(selected)))]
    if not selected and sorted_items:
        selected = [sorted_items[0][0]]
    ordered_selected = [c for c in ordered_features if c in selected]
    dropped = [c for c in ordered_features if c not in ordered_selected]
    return ordered_selected, dropped


class VarianceThresholdSelector(Transformer):
    """Remove low-variance features from a Polars DataFrame."""

    def __init__(
        self,
        threshold: float = 0.0,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.columns = None if columns is None else list(columns)
        self.variances_: Dict[str, float] = {}
        self.selected_features_: List[str] = []
        self.dropped_features_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, df: pl.DataFrame) -> "VarianceThresholdSelector":
        df = _ensure_polars_df(df)
        cols = _resolve_numeric_columns(df, self.columns)
        self.feature_names_in_ = list(cols)
        self.variances_.clear()
        selected: List[str] = []
        dropped: List[str] = []
        for column in cols:
            series = df.get_column(column).cast(pl.Float64)
            clean = series.drop_nulls()
            if clean.is_empty():
                variance = 0.0
            else:
                variance = float(clean.var())
                if math.isnan(variance):
                    variance = 0.0
            self.variances_[column] = variance
            if variance > self.threshold:
                selected.append(column)
            else:
                dropped.append(column)
        self.selected_features_ = selected
        self.dropped_features_ = dropped
        if dropped:
            self.feature_names_out_ = [c for c in df.columns if c not in dropped]
        else:
            self.feature_names_out_ = list(df.columns)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        df = _ensure_polars_df(df)
        to_drop = [c for c in self.dropped_features_ if c in df.columns]
        if not to_drop:
            return df
        return df.drop(to_drop)

    def get_support(self) -> List[bool]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before get_support")
        cols = self.feature_names_in_ or []
        selected = set(self.selected_features_)
        return [c in selected for c in cols]

    def get_selected_features(self) -> List[str]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before get_selected_features")
        return list(self.selected_features_)


class MutualInformationSelector(Transformer):
    """Select features using mutual information against a target column."""

    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        k: Optional[int] = None,
        threshold: Optional[float] = 0.0,
        problem: str = "auto",
        random_state: Optional[int] = 0,
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.k = k
        self.threshold = threshold
        self.problem = problem
        self.random_state = random_state

        self.problem_: Optional[str] = None
        self.mutual_information_: Dict[str, float] = {}
        self.selected_features_: List[str] = []
        self.dropped_features_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, df: pl.DataFrame) -> "MutualInformationSelector":
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found")

        df = _ensure_polars_df(df)
        cols = _resolve_numeric_columns(df, self.columns)
        if self.target in cols:
            cols.remove(self.target)
        if not cols:
            raise ValueError("No feature columns available for selection")

        target_series = df.get_column(self.target)
        problem = _infer_problem_type(target_series, self.problem)
        self.problem_ = problem

        train_df = _prepare_training_frame(df, cols, self.target)
        if train_df.is_empty():
            raise ValueError("No rows available after dropping nulls for mutual information computation")

        X = train_df.select(cols).to_numpy()
        y_series = train_df.get_column(self.target)

        try:
            if problem == "classification":
                from sklearn.feature_selection import mutual_info_classif
                from sklearn.preprocessing import LabelEncoder

                encoder = LabelEncoder()
                y = encoder.fit_transform(y_series.to_list())
                scores = mutual_info_classif(
                    X,
                    y,
                    random_state=self.random_state,
                )
            else:
                from sklearn.feature_selection import mutual_info_regression

                y = y_series.to_numpy()
                scores = mutual_info_regression(
                    X,
                    y,
                    random_state=self.random_state,
                )
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "MutualInformationSelector requires scikit-learn. Install with `pip install scikit-learn`."
            ) from exc

        mi_scores = {col: float(score) for col, score in zip(cols, scores)}
        self.mutual_information_ = mi_scores

        selected, dropped = _select_features(mi_scores, cols, self.k, self.threshold)
        self.selected_features_ = selected
        self.dropped_features_ = dropped
        if dropped:
            self.feature_names_out_ = [c for c in df.columns if c not in dropped]
        else:
            self.feature_names_out_ = list(df.columns)
        self.feature_names_in_ = list(cols)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        df = _ensure_polars_df(df)
        to_drop = [c for c in self.dropped_features_ if c in df.columns]
        if not to_drop:
            return df
        return df.drop(to_drop)

    def get_support(self) -> List[bool]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before get_support")
        cols = self.feature_names_in_ or []
        selected = set(self.selected_features_)
        return [c in selected for c in cols]

    def get_selected_features(self) -> List[str]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before get_selected_features")
        return list(self.selected_features_)


class ModelBasedImportanceSelector(Transformer):
    """Select features using importances from a supervised estimator (requires scikit-learn)."""

    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        estimator_factory: Optional[Callable[[], object]] = None,
        importance_getter: str = "auto",
        k: Optional[int] = None,
        threshold: Optional[float] = 0.0,
        problem: str = "auto",
        random_state: Optional[int] = 0,
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.estimator_factory = estimator_factory
        self.importance_getter = importance_getter
        self.k = k
        self.threshold = threshold
        self.problem = problem
        self.random_state = random_state

        self.problem_: Optional[str] = None
        self.estimator_: Optional[object] = None
        self.feature_importances_: Dict[str, float] = {}
        self.selected_features_: List[str] = []
        self.dropped_features_: List[str] = []
        self.feature_names_out_: List[str] = []

    def _default_estimator(self, problem: str):  # pragma: no cover - simple factory
        try:
            if problem == "classification":
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(
                    n_estimators=200,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(
                    n_estimators=200,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
        except ImportError as exc:
            raise ImportError(
                "ModelBasedImportanceSelector requires scikit-learn. Install with `pip install scikit-learn`."
            ) from exc

    def _get_importances(self, estimator: object, cols: Sequence[str]) -> Dict[str, float]:
        getter = self.importance_getter
        if getter == "auto":
            if hasattr(estimator, "feature_importances_"):
                getter = "feature_importances_"
            elif hasattr(estimator, "coef_"):
                getter = "coef_"
            else:
                raise AttributeError(
                    "Provided estimator does not expose feature importances or coefficients"
                )

        values = getattr(estimator, getter)
        array = np.asarray(values)
        if array.ndim > 1:
            array = np.linalg.norm(array, axis=0)
        return {col: float(score) for col, score in zip(cols, array.tolist())}

    def fit(self, df: pl.DataFrame) -> "ModelBasedImportanceSelector":
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found")

        df = _ensure_polars_df(df)
        cols = _resolve_numeric_columns(df, self.columns)
        if self.target in cols:
            cols.remove(self.target)
        if not cols:
            raise ValueError("No feature columns available for selection")

        target_series = df.get_column(self.target)
        problem = _infer_problem_type(target_series, self.problem)
        self.problem_ = problem

        train_df = _prepare_training_frame(df, cols, self.target)
        if train_df.is_empty():
            raise ValueError("No rows available after dropping nulls for model-based selection")

        X = train_df.select(cols).to_numpy()
        y_series = train_df.get_column(self.target)

        try:
            if problem == "classification":
                from sklearn.preprocessing import LabelEncoder

                encoder = LabelEncoder()
                y = encoder.fit_transform(y_series.to_list())
            else:
                y = y_series.to_numpy()

            estimator = (
                self.estimator_factory()
                if self.estimator_factory is not None
                else self._default_estimator(problem)
            )
            fit_estimator = estimator.fit(X, y)
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "ModelBasedImportanceSelector requires scikit-learn. Install with `pip install scikit-learn`."
            ) from exc

        importances = self._get_importances(fit_estimator, cols)
        self.feature_importances_ = importances

        selected, dropped = _select_features(importances, cols, self.k, self.threshold)
        self.selected_features_ = selected
        self.dropped_features_ = dropped
        self.estimator_ = fit_estimator
        if dropped:
            self.feature_names_out_ = [c for c in df.columns if c not in dropped]
        else:
            self.feature_names_out_ = list(df.columns)
        self.feature_names_in_ = list(cols)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        df = _ensure_polars_df(df)
        to_drop = [c for c in self.dropped_features_ if c in df.columns]
        if not to_drop:
            return df
        return df.drop(to_drop)

    def get_support(self) -> List[bool]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before get_support")
        cols = self.feature_names_in_ or []
        selected = set(self.selected_features_)
        return [c in selected for c in cols]

    def get_selected_features(self) -> List[str]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before get_selected_features")
        return list(self.selected_features_)


class MRMRSelector(Transformer):
    """Minimum Redundancy Maximum Relevance selection leveraging mutual information."""

    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        k: Optional[int] = None,
        problem: str = "auto",
        random_state: Optional[int] = 0,
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.k = k
        self.problem = problem
        self.random_state = random_state

        self.problem_: Optional[str] = None
        self.mutual_information_: Dict[str, float] = {}
        self.mrmr_scores_: Dict[str, float] = {}
        self.selected_features_: List[str] = []
        self.dropped_features_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, df: pl.DataFrame) -> "MRMRSelector":
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found")

        df = _ensure_polars_df(df)
        cols = _resolve_numeric_columns(df, self.columns)
        if self.target in cols:
            cols.remove(self.target)
        if not cols:
            raise ValueError("No feature columns available for selection")

        target_series = df.get_column(self.target)
        problem = _infer_problem_type(target_series, self.problem)
        self.problem_ = problem

        train_df = _prepare_training_frame(df, cols, self.target)
        if train_df.is_empty():
            raise ValueError("No rows available after dropping nulls for MRMR computation")

        X_df = train_df.select(cols)
        X = X_df.to_numpy()
        y_series = train_df.get_column(self.target)

        try:
            if problem == "classification":
                from sklearn.feature_selection import mutual_info_classif
                from sklearn.preprocessing import LabelEncoder

                encoder = LabelEncoder()
                y = encoder.fit_transform(y_series.to_list())
                mi_scores = mutual_info_classif(
                    X,
                    y,
                    random_state=self.random_state,
                )
            else:
                from sklearn.feature_selection import mutual_info_regression

                y = y_series.to_numpy()
                mi_scores = mutual_info_regression(
                    X,
                    y,
                    random_state=self.random_state,
                )
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "MRMRSelector requires scikit-learn. Install with `pip install scikit-learn`."
            ) from exc

        mi_map = {col: float(score) for col, score in zip(cols, mi_scores)}
        self.mutual_information_ = mi_map

        corr_matrix = np.corrcoef(X, rowvar=False)
        if np.isnan(corr_matrix).any():
            corr_matrix = np.nan_to_num(corr_matrix)

        feature_indices = {col: idx for idx, col in enumerate(cols)}
        remaining = set(cols)
        selected: List[str] = []
        mrmr_scores: Dict[str, float] = {}

        k = self.k if self.k is not None else len(cols)
        k = max(1, min(k, len(cols)))

        first = max(remaining, key=lambda c: mi_map[c])
        selected.append(first)
        remaining.remove(first)
        mrmr_scores[first] = mi_map[first]

        while remaining and len(selected) < k:
            best_feature = None
            best_score = -float("inf")
            for candidate in list(remaining):
                idx = feature_indices[candidate]
                redundancy = 0.0
                if selected:
                    sel_indices = [feature_indices[s] for s in selected]
                    redundancy = float(np.mean(np.abs(corr_matrix[idx, sel_indices])))
                score = mi_map[candidate] - redundancy
                if score > best_score:
                    best_score = score
                    best_feature = candidate
            if best_feature is None:
                break
            selected.append(best_feature)
            remaining.remove(best_feature)
            mrmr_scores[best_feature] = best_score

        selected, _ = _select_features(mrmr_scores, cols, len(selected), None)
        self.selected_features_ = selected
        self.dropped_features_ = [c for c in cols if c not in selected]
        self.mrmr_scores_ = {col: mrmr_scores.get(col, 0.0) for col in cols}
        if self.dropped_features_:
            self.feature_names_out_ = [c for c in df.columns if c not in self.dropped_features_]
        else:
            self.feature_names_out_ = list(df.columns)
        self.feature_names_in_ = list(cols)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        df = _ensure_polars_df(df)
        to_drop = [c for c in self.dropped_features_ if c in df.columns]
        if not to_drop:
            return df
        return df.drop(to_drop)

    def get_support(self) -> List[bool]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before get_support")
        cols = self.feature_names_in_ or []
        selected = set(self.selected_features_)
        return [c in selected for c in cols]

    def get_selected_features(self) -> List[str]:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before get_selected_features")
        return list(self.selected_features_)
