from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import polars as pl
import numpy as np

from .base import Transformer, _ensure_polars_df
from .encoders import (
    OneHotEncoder,
    CountFrequencyEncoder,
    RareLabelEncoder,
    MeanEncoder,
    WeightOfEvidenceEncoder,
    LeaveOneOutEncoder,
)
from .scale import RobustScaler, QuantileRankTransformer
from .impute import SimpleImputer
from .outliers import ClipOutliers
from .discretize import EqualFrequencyDiscretizer
from .vst import LogTransformer, YeoJohnsonTransformer
from .snapshot import TimeSnapshotAggregator
from .interaction import FeatureInteractions


def _is_string(dtype: pl.DataType) -> bool:
    try:
        return bool(dtype.is_(pl.Utf8) or dtype.is_(pl.String))  # type: ignore[attr-defined]
    except AttributeError:
        return False


def _numeric_columns(df: pl.DataFrame) -> List[str]:
    cols: List[str] = []
    for name, dt in zip(df.columns, df.dtypes):
        try:
            if dt.is_numeric():  # type: ignore[attr-defined]
                cols.append(name)
        except AttributeError:
            # conservative: skip if unknown
            pass
    return cols


def _categorical_columns(df: pl.DataFrame) -> List[str]:
    cols: List[str] = []
    for name, dt in zip(df.columns, df.dtypes):
        if _is_string(dt) or dt == pl.Categorical:
            cols.append(name)
    return cols


class AutoFeaturizer(Transformer):
    """Automatically pick simple, robust feature engineering per column.

    Strategy
    - Categorical (strings/categorical):
        * if unique values <= max_ohe_cardinality -> OneHotEncoder
        * else -> CountFrequencyEncoder(normalize=True)
    - Boolean: treated as low-card categorical; included in OneHotEncoder set (with drop_first=True)
    - Numeric (non-boolean): scaled via RobustScaler (median/MAD)
    - Optional numeric ranks via QuantileRankTransformer
    - Optional outlier clipping via ClipOutliers
    - Optional variance-stabilizing transforms on skewed columns (log / Yeo-Johnson)
    - Optional discretization via equal-frequency bins
    - Optional time-window aggregations and group-based feature interactions
    - Optional supervised encoders (Mean / Weight-of-Evidence / Leave-One-Out) when a target column is provided

    Parameters
    - max_ohe_cardinality: threshold for using one-hot vs frequency encoding
    - add_ranks: also add QuantileRankTransformer per numeric column
    - drop_original: whether sub-transformers drop their source columns
    """

    def __init__(
        self,
        max_ohe_cardinality: int = 10,
        add_ranks: bool = False,
        drop_original: bool = True,
        # optional steps
        include_imputation: bool = True,
        include_outliers: bool = False,
        include_vst: bool = True,
        include_discretization: bool = False,
        discretize_bins: int = 5,
        skew_threshold: float = 1.0,
        # time & interactions (opt-in)
        time_column: Optional[str] = None,
        time_windows: Sequence[str] = ("30d",),
        groupby: Optional[Sequence[str]] = None,
        interaction_aggs: Sequence[str] = ("mean",),
        # supervised
        target_column: Optional[str] = None,
        include_target_encoders: bool = True,
    ) -> None:
        self.max_ohe_cardinality = int(max_ohe_cardinality)
        self.add_ranks = bool(add_ranks)
        self.drop_original = bool(drop_original)

        self.include_imputation = bool(include_imputation)
        self.include_outliers = bool(include_outliers)
        self.include_vst = bool(include_vst)
        self.include_discretization = bool(include_discretization)
        self.discretize_bins = int(discretize_bins)
        self.skew_threshold = float(skew_threshold)

        self.time_column = time_column
        self.time_windows = list(time_windows)
        self.groupby = None if groupby is None else list(groupby)
        self.interaction_aggs = list(interaction_aggs)

        self.target_column = target_column
        self.include_target_encoders = bool(include_target_encoders)

        self.pipeline_: List[Transformer] = []
        self.plan_: Dict[str, Dict[str, List[str]]] = {}
        self.numeric_cols_: List[str] = []

    def fit(self, df: pl.DataFrame) -> "AutoFeaturizer":
        df = _ensure_polars_df(df)

        groupby_cols = set(self.groupby or [])

        cat_cols = _categorical_columns(df)
        bool_cols = [n for n, dt in zip(df.columns, df.dtypes) if dt == pl.Boolean]
        num_cols = [c for c in _numeric_columns(df) if c not in bool_cols]

        if self.target_column and self.target_column in df.columns:
            cat_cols = [c for c in cat_cols if c != self.target_column]
            bool_cols = [c for c in bool_cols if c != self.target_column]
            num_cols = [c for c in num_cols if c != self.target_column]

        self.numeric_cols_ = list(num_cols)

        # decide categorical encoding sets
        ohe_cols: List[str] = []
        freq_cols: List[str] = []
        for c in cat_cols + bool_cols:
            try:
                n_unique = int(df.get_column(c).drop_nulls().n_unique())
            except Exception:
                n_unique = self.max_ohe_cardinality + 1
            if n_unique <= self.max_ohe_cardinality:
                ohe_cols.append(c)
            else:
                freq_cols.append(c)

        pipeline: List[Transformer] = []
        plan: Dict[str, Dict[str, List[str]]] = {
            "categorical": {},
            "numeric": {},
            "missing": {},
            "outliers": {},
            "time": {},
            "interactions": {},
            "target": {},
        }

        # Imputation (simple) if requested
        if self.include_imputation:
            miss_num = [c for c in num_cols if df.get_column(c).null_count() > 0]
            miss_cat = [c for c in (cat_cols + bool_cols) if df.get_column(c).null_count() > 0]
            impute_cols = list(set(miss_num + miss_cat))
            if impute_cols:
                pipeline.append(
                    SimpleImputer(
                        columns=impute_cols,
                        numerical_strategy="median",
                        categorical_strategy="most_frequent",
                        add_indicator=True,
                        drop_original=self.drop_original,
                    )
                )
                plan["missing"]["simple_imputer"] = list(impute_cols)

        # Target-based encoders (supervised)
        if (
            self.include_target_encoders
            and self.target_column
            and self.target_column in df.columns
        ):
            target_series = df.get_column(self.target_column)
            try:
                target_unique = int(target_series.drop_nulls().n_unique())
            except Exception:
                target_unique = 0
            try:
                is_numeric_target = bool(target_series.dtype.is_numeric())  # type: ignore[attr-defined]
            except AttributeError:
                is_numeric_target = target_series.dtype in {
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                    pl.Float32,
                    pl.Float64,
                }
            cat_for_target = [c for c in (cat_cols + bool_cols) if c != self.target_column]
            if cat_for_target:
                target_keep = [c for c in cat_for_target if c in groupby_cols]
                target_drop = [c for c in cat_for_target if c not in groupby_cols]

                encoder_factory: Callable[[Sequence[str], bool], Transformer] | None
                plan_key: str | None
                if is_numeric_target and target_unique > 2:
                    encoder_factory = lambda cols, drop: MeanEncoder(  # type: ignore[assignment]
                        target=self.target_column,
                        columns=cols,
                        smoothing=5.0,
                        drop_original=drop,
                    )
                    plan_key = "mean_encoder"
                else:
                    if target_unique == 2:
                        encoder_factory = lambda cols, drop: WeightOfEvidenceEncoder(  # type: ignore[assignment]
                            target=self.target_column,
                            columns=cols,
                            drop_original=drop,
                        )
                        plan_key = "weight_of_evidence"
                    else:
                        encoder_factory = lambda cols, drop: LeaveOneOutEncoder(  # type: ignore[assignment]
                            target=self.target_column,
                            columns=cols,
                            smoothing=5.0,
                            drop_original=drop,
                        )
                        plan_key = "leave_one_out"

                if encoder_factory and plan_key:
                    encoded: List[str] = []
                    for cols, drop in ((target_drop, self.drop_original), (target_keep, False)):
                        if cols:
                            pipeline.append(encoder_factory(cols, drop))
                            encoded.extend(cols)
                    if encoded:
                        plan["target"][plan_key] = list(encoded)

                ohe_cols = [c for c in ohe_cols if c not in cat_for_target]
                freq_cols = [c for c in freq_cols if c not in cat_for_target]

        if ohe_cols:
            drop_cols = [c for c in ohe_cols if c not in groupby_cols]
            keep_cols = [c for c in ohe_cols if c in groupby_cols]
            encoded: List[str] = []
            if drop_cols:
                pipeline.append(
                    OneHotEncoder(columns=drop_cols, drop_first=True, drop_original=self.drop_original)
                )
                encoded.extend(drop_cols)
            if keep_cols:
                pipeline.append(OneHotEncoder(columns=keep_cols, drop_first=True, drop_original=False))
                encoded.extend(keep_cols)
            if encoded:
                plan["categorical"]["one_hot"] = list(encoded)
        if freq_cols:
            drop_cols = [c for c in freq_cols if c not in groupby_cols]
            keep_cols = [c for c in freq_cols if c in groupby_cols]
            high_card: List[str] = []
            for c in drop_cols:
                try:
                    n_unique = int(df.get_column(c).drop_nulls().n_unique())
                except Exception:
                    n_unique = self.max_ohe_cardinality * 5
                if n_unique > self.max_ohe_cardinality * 5:
                    high_card.append(c)
            if high_card:
                pipeline.append(RareLabelEncoder(columns=high_card, min_frequency=0.01, drop_original=False))
                plan["categorical"]["rare_label"] = list(high_card)
            encoded: List[str] = []
            if drop_cols:
                pipeline.append(
                    CountFrequencyEncoder(columns=drop_cols, normalize=True, drop_original=self.drop_original)
                )
                encoded.extend(drop_cols)
            if keep_cols:
                pipeline.append(CountFrequencyEncoder(columns=keep_cols, normalize=True, drop_original=False))
                encoded.extend(keep_cols)
            if encoded:
                plan["categorical"]["count_freq"] = list(encoded)

        if num_cols:
            # Outlier handling (optional) before other numeric transforms
            if self.include_outliers:
                pipeline.append(ClipOutliers(columns=num_cols, method="iqr", action="clip"))
                plan["outliers"]["clip_iqr"] = list(num_cols)

            # Variance-stabilizing transforms for skew
            if self.include_vst:
                pos_cols: List[str] = []
                any_cols: List[str] = []
                for c in num_cols:
                    s = df.get_column(c).cast(pl.Float64)
                    arr = np.array(s.to_list(), dtype=float)
                    if arr.size < 3 or np.nanstd(arr) == 0:
                        continue
                    mu = float(np.nanmean(arr))
                    sd = float(np.nanstd(arr))
                    if sd == 0:
                        continue
                    skew = float(np.nanmean(((arr - mu) / (sd if sd != 0 else 1.0)) ** 3))
                    if abs(skew) >= self.skew_threshold:
                        if np.nanmin(arr) > 0:
                            pos_cols.append(c)
                        else:
                            any_cols.append(c)
                if pos_cols:
                    pipeline.append(LogTransformer(columns=pos_cols, drop_original=False))
                    plan["numeric"]["log_transform"] = list(pos_cols)
                if any_cols:
                    pipeline.append(YeoJohnsonTransformer(columns=any_cols, drop_original=False))
                    plan["numeric"]["yeojohnson"] = list(any_cols)

            # Discretization (optional)
            if self.include_discretization:
                disc_cols = [
                    c
                    for c in num_cols
                    if df.get_column(c).drop_nulls().n_unique() > (self.discretize_bins * 2)
                ]
                if disc_cols:
                    pipeline.append(
                        EqualFrequencyDiscretizer(
                            columns=disc_cols,
                            n_bins=self.discretize_bins,
                            drop_original=False,
                        )
                    )
                    plan["numeric"]["discretize_equal_freq"] = list(disc_cols)

            if self.add_ranks:
                pipeline.append(QuantileRankTransformer(columns=num_cols, drop_original=False))
                plan["numeric"]["ranks"] = list(num_cols)

            pipeline.append(RobustScaler(columns=num_cols, drop_original=False))
            plan["numeric"]["robust_scale"] = list(num_cols)

        # Time snapshot features (opt-in)
        if self.time_column and self.time_column in df.columns:
            val_cols = [c for c in num_cols if c != self.time_column]
            if val_cols:
                pipeline.append(
                    TimeSnapshotAggregator(
                        time_column=self.time_column,
                        groupby=self.groupby,
                        value_columns=val_cols,
                        windows=self.time_windows,
                        aggregations=("mean",),
                        include_current=True,
                        drop_original=False,
                    )
                )
                plan["time"]["snapshot"] = [self.time_column]

        # Interactions (opt-in)
        if self.groupby:
            value_cols = list(num_cols)
            if value_cols:
                pipeline.append(
                    FeatureInteractions(groupby=self.groupby, value_columns=value_cols, aggregations=self.interaction_aggs, drop_original=False)
                )
                plan["interactions"]["group_aggs"] = list(self.groupby)

        # store & fit underlying transformers
        self.pipeline_ = pipeline
        self.plan_ = plan
        self.feature_names_in_ = list(df.columns)
        for t in self.pipeline_:
            t.fit(df)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = _ensure_polars_df(df)
        for t in self.pipeline_:
            out = t.transform(out)
        if self.drop_original and self.numeric_cols_:
            keep_cols = set(self.groupby or [])
            if self.time_column:
                keep_cols.add(self.time_column)
            if self.target_column:
                keep_cols.add(self.target_column)
            drop_cols = [c for c in self.numeric_cols_ if c not in keep_cols and c in out.columns]
            if drop_cols:
                out = out.drop(drop_cols)
        return out

    # Convenience
    def get_plan(self) -> Dict[str, Dict[str, List[str]]]:
        return {k: {kk: list(vv) for kk, vv in v.items()} for k, v in self.plan_.items()}


def suggest_methods(
    df: pl.DataFrame,
    max_ohe_cardinality: int = 10,
    include_imputation: bool = True,
    include_outliers: bool = False,
    include_vst: bool = True,
    include_discretization: bool = False,
    discretize_bins: int = 5,
    skew_threshold: float = 1.0,
    time_column: Optional[str] = None,
    groupby: Optional[Sequence[str]] = None,
    target_column: Optional[str] = None,
    include_target_encoders: bool = True,
) -> Dict[str, Dict[str, List[str]]]:
    """Return a plan of suggested methods per column without fitting anything.

    See AutoFeaturizer for the selection logic. This helper is convenient when you
    just want the mapping of columns -> methods.
    """
    df = _ensure_polars_df(df)
    cat_cols = _categorical_columns(df)
    bool_cols = [n for n, dt in zip(df.columns, df.dtypes) if dt == pl.Boolean]
    num_cols = [c for c in _numeric_columns(df) if c not in bool_cols]

    if target_column and target_column in df.columns:
        cat_cols = [c for c in cat_cols if c != target_column]
        bool_cols = [c for c in bool_cols if c != target_column]
        num_cols = [c for c in num_cols if c != target_column]

    ohe_cols: List[str] = []
    freq_cols: List[str] = []
    for c in cat_cols + bool_cols:
        try:
            n_unique = int(df.get_column(c).drop_nulls().n_unique())
        except Exception:
            n_unique = max_ohe_cardinality + 1
        if n_unique <= max_ohe_cardinality:
            ohe_cols.append(c)
        else:
            freq_cols.append(c)

    plan: Dict[str, Dict[str, List[str]]] = {
        "categorical": {"one_hot": ohe_cols, "count_freq": freq_cols},
        "numeric": {"robust_scale": num_cols},
        "missing": {},
        "outliers": {},
        "time": {},
        "interactions": {},
        "target": {},
    }

    if include_imputation:
        miss_cols = [c for c in df.columns if df.get_column(c).null_count() > 0]
        if miss_cols:
            plan["missing"]["simple_imputer"] = miss_cols

    if include_outliers:
        plan["outliers"]["clip_iqr"] = list(num_cols)

    if include_vst:
        pos_cols: List[str] = []
        any_cols: List[str] = []
        for c in num_cols:
            arr = np.array(df.get_column(c).cast(pl.Float64).to_list(), dtype=float)
            if arr.size < 3 or np.nanstd(arr) == 0:
                continue
            mu = float(np.nanmean(arr))
            sd = float(np.nanstd(arr))
            if sd == 0:
                continue
            skew = float(np.nanmean(((arr - mu) / (sd if sd != 0 else 1.0)) ** 3))
            if abs(skew) >= skew_threshold:
                if np.nanmin(arr) > 0:
                    pos_cols.append(c)
                else:
                    any_cols.append(c)
        if pos_cols:
            plan["numeric"]["log_transform"] = pos_cols
        if any_cols:
            plan["numeric"]["yeojohnson"] = any_cols

    if include_discretization:
        disc_cols = [c for c in num_cols if df.get_column(c).drop_nulls().n_unique() > (discretize_bins * 2)]
        if disc_cols:
            plan["numeric"]["discretize_equal_freq"] = disc_cols

    if time_column and time_column in df.columns and num_cols:
        plan["time"]["snapshot"] = [time_column]

    if groupby:
        plan["interactions"]["group_aggs"] = list(groupby)

    if (
        include_target_encoders
        and target_column
        and target_column in df.columns
    ):
        target_series = df.get_column(target_column)
        try:
            target_unique = int(target_series.drop_nulls().n_unique())
        except Exception:
            target_unique = 0
        try:
            is_numeric_target = bool(target_series.dtype.is_numeric())  # type: ignore[attr-defined]
        except AttributeError:
            is_numeric_target = target_series.dtype in {
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.Float32,
                pl.Float64,
            }
        cat_for_target = [c for c in (cat_cols + bool_cols) if c != target_column]
        if cat_for_target:
            if is_numeric_target and target_unique > 2:
                plan["target"]["mean_encoder"] = cat_for_target
            else:
                if target_unique == 2:
                    plan["target"]["weight_of_evidence"] = cat_for_target
                else:
                    plan["target"]["leave_one_out"] = cat_for_target
            # remove supervised columns from categorical suggestions
            plan["categorical"]["one_hot"] = [c for c in plan["categorical"].get("one_hot", []) if c not in cat_for_target]
            plan["categorical"]["count_freq"] = [c for c in plan["categorical"].get("count_freq", []) if c not in cat_for_target]

    return plan
