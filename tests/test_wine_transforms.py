import pytest

pl = pytest.importorskip("polars", reason="polars is required for feataz tests")

if not hasattr(pl.datatypes, "is_numeric"):
    def _is_numeric(dtype: pl.DataType) -> bool:  # type: ignore[name-defined]
        try:
            return bool(dtype.is_numeric())  # type: ignore[attr-defined]
        except AttributeError:
            return False

    pl.datatypes.is_numeric = _is_numeric  # type: ignore[attr-defined]

if not hasattr(pl.datatypes, "is_string_dtype"):
    def _is_string(dtype: pl.DataType) -> bool:  # type: ignore[name-defined]
        try:
            return bool(dtype.is_(pl.Utf8) or dtype.is_(pl.String))  # type: ignore[attr-defined]
        except AttributeError:
            return False

    pl.datatypes.is_string_dtype = _is_string  # type: ignore[attr-defined]

from feataz.encoders import (
    BinaryEncoder,
    CountFrequencyEncoder,
    HashEncoder,
    LeaveOneOutEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
    StringSimilarityEncoder,
    WeightOfEvidenceEncoder,
)
from feataz.discretize import (
    ArbitraryDiscretizer,
    EqualFrequencyDiscretizer,
    EqualWidthDiscretizer,
    GeometricWidthDiscretizer,
)
from feataz.features import (
    CyclicalFeatures,
    MathFeatures,
    RelativeFeatures,
)
from feataz.interaction import FeatureInteractions
from feataz.impute import GroupImputer, SimpleImputer
from feataz.outliers import ClipOutliers
from feataz.scale import QuantileRankTransformer, RobustScaler
from feataz.snapshot import (
    DynamicSnapshotAggregator,
    EWMAggregator,
    TimeSnapshotAggregator,
    ToDateSnapshotAggregator,
)
from feataz.auto import AutoFeaturizer, suggest_methods

try:  # optional dependency: numpy
    from feataz.vst import (
        ArcsinTransformer,
        BoxCoxTransformer,
        LogCPTransformer,
        LogTransformer,
        PowerTransformer,
        ReciprocalTransformer,
        YeoJohnsonTransformer,
    )
except Exception:  # pragma: no cover - skip if numpy missing
    VST_AVAILABLE = False
else:
    VST_AVAILABLE = True


def _ensure_new_columns(original: pl.DataFrame, updated: pl.DataFrame) -> None:
    new_cols = sorted(set(updated.columns) - set(original.columns))
    assert new_cols, "expected transformer to add new columns"
    for name in new_cols:
        assert updated.get_column(name).null_count() < updated.height


def test_encoders_transform_wine(wine_df: pl.DataFrame) -> None:
    df = wine_df.with_columns(
        (pl.col("target") == 0).cast(pl.Int64).alias("target_bin"),
        pl.when(pl.col("magnesium") < 100)
        .then(pl.lit("low"))
        .when(pl.col("magnesium") < 130)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
        .alias("magnesium_band"),
    )

    encoders = [
        OneHotEncoder(columns=["target_str"], drop_original=False),
        OrdinalEncoder(columns=["target_str"], drop_original=False),
        CountFrequencyEncoder(columns=["target_str"], normalize=True, drop_original=False),
        MeanEncoder(target="target_bin", columns=["target_str"], smoothing=5.0, drop_original=False),
        WeightOfEvidenceEncoder(target="target_bin", columns=["target_str"], multi_class="binary", drop_original=False),
        RareLabelEncoder(columns=["magnesium_band"], min_frequency=0.2, drop_original=False),
        StringSimilarityEncoder(columns=["target_str"], top_k_anchors=2, drop_original=False),
        HashEncoder(columns=["target_str"], n_components=16, drop_original=False),
        BinaryEncoder(columns=["target_str"], drop_original=False),
        LeaveOneOutEncoder(target="target_bin", columns=["target_str"], smoothing=2.0, drop_original=False),
    ]

    for encoder in encoders:
        transformed = encoder.fit_transform(df)
        assert transformed.height == df.height
        _ensure_new_columns(df, transformed)


def test_weight_of_evidence_boolean_target(wine_df: pl.DataFrame) -> None:
    df = wine_df.with_columns(
        (pl.col("target") == 0).alias("target_bool")
    )
    encoder = WeightOfEvidenceEncoder(
        target="target_bool",
        columns=["target_str"],
        multi_class="binary",
        drop_original=False,
    )
    transformed = encoder.fit_transform(df)
    woe_col = "target_str__woe"
    fallback = encoder.defaults_["target_str"][woe_col]
    assert transformed.select(pl.col(woe_col).n_unique()).item() > 1
    assert transformed.filter(pl.col(woe_col) != fallback).height > 0


def test_weight_of_evidence_string_binary_target(wine_df: pl.DataFrame) -> None:
    df = wine_df.with_columns(
        pl.when(pl.col("target") == 0)
        .then(pl.lit("zero"))
        .otherwise(pl.lit("non_zero"))
        .alias("target_str_bin")
    )
    encoder = WeightOfEvidenceEncoder(
        target="target_str_bin",
        columns=["target_str"],
        multi_class="binary",
        drop_original=False,
    )
    transformed = encoder.fit_transform(df)
    woe_col = "target_str__woe"
    fallback = encoder.defaults_["target_str"][woe_col]
    assert transformed.select(pl.col(woe_col).n_unique()).item() > 1
    assert transformed.filter(pl.col(woe_col) != fallback).height > 0


def test_discretizers_transform_wine(wine_df: pl.DataFrame) -> None:
    df = wine_df.select(["alcohol", "malic_acid"])
    discretizers = [
        EqualFrequencyDiscretizer(columns=["alcohol", "malic_acid"], n_bins=4, drop_original=False),
        EqualWidthDiscretizer(columns=["alcohol", "malic_acid"], n_bins=4, drop_original=False),
        GeometricWidthDiscretizer(columns=["alcohol", "malic_acid"], n_bins=4, drop_original=False),
        ArbitraryDiscretizer(bins={"alcohol": [-float("inf"), 12.0, 13.5, float("inf")]}, drop_original=False),
    ]

    for disc in discretizers:
        transformed = disc.fit_transform(df)
        assert transformed.height == df.height
        _ensure_new_columns(df, transformed)


@pytest.mark.skipif(not VST_AVAILABLE, reason="numpy is required for variance-stabilizing transformers")
def test_vst_transformers_wine(wine_df: pl.DataFrame) -> None:
    df = wine_df.select(["alcohol", "malic_acid"]).with_columns(
        (pl.col("alcohol") / pl.col("alcohol").max()).alias("alcohol_unit")
    )

    transformers = [
        LogTransformer(columns=["alcohol"], drop_original=False),
        LogCPTransformer(columns=["alcohol"], c=1.0, drop_original=False),
        ReciprocalTransformer(columns=["alcohol"], drop_original=False),
        PowerTransformer(power=0.5, columns=["alcohol"], drop_original=False),
        BoxCoxTransformer(columns=["alcohol"], drop_original=False),
        YeoJohnsonTransformer(columns=["malic_acid"], drop_original=False),
        ArcsinTransformer(columns=["alcohol_unit"], scale=False, drop_original=False),
    ]

    for transformer in transformers:
        transformed = transformer.fit_transform(df)
        assert transformed.height == df.height
        _ensure_new_columns(df, transformed)


def test_feature_generators_wine(wine_df: pl.DataFrame) -> None:
    df = wine_df.with_columns(
        (pl.col("index") % 24).cast(pl.Float64).alias("hour"),
        (pl.col("target") == 0).cast(pl.Int64).alias("target_bin"),
    )

    math = MathFeatures(columns=["alcohol", "malic_acid"], powers=[2], drop_original=False)
    math_out = math.fit_transform(df)
    assert math_out.height == df.height
    _ensure_new_columns(df, math_out)

    rel = RelativeFeatures(target_columns=["alcohol"], reference_columns=["malic_acid"], drop_original=False)
    rel_out = rel.fit_transform(df)
    assert rel_out.height == df.height
    _ensure_new_columns(df, rel_out)

    cyc = CyclicalFeatures(columns=["hour"], period=24.0, drop_original=False)
    cyc_out = cyc.fit_transform(df)
    assert cyc_out.height == df.height
    _ensure_new_columns(df, cyc_out)

    interactions = FeatureInteractions(
        groupby=["target_str"],
        value_columns=["alcohol"],
        aggregations=["mean", "sum"],
        drop_original=False,
    )
    inter_out = interactions.fit_transform(df)
    assert inter_out.height == df.height
    _ensure_new_columns(df, inter_out)


def test_snapshot_aggregators_wine(wine_with_time: pl.DataFrame) -> None:
    df = wine_with_time.select(["ts", "target_str", "alcohol", "color_intensity"]).with_columns(
        (pl.col("target_str") == "class_0").cast(pl.Int8).alias("group_flag")
    )

    time_snap = TimeSnapshotAggregator(
        time_column="ts",
        groupby=["target_str"],
        value_columns=["alcohol"],
        windows=["30d"],
        aggregations=["mean"],
        include_current=True,
        drop_original=False,
    )
    time_out = time_snap.fit_transform(df)
    assert time_out.height == df.height
    _ensure_new_columns(df, time_out)

    dynamic = DynamicSnapshotAggregator(
        time_column="ts",
        every="30d",
        period="30d",
        groupby=["target_str"],
        value_columns=["alcohol"],
        aggregations=["sum"],
        drop_original=False,
    )
    dyn_out = dynamic.fit_transform(df)
    assert dyn_out.height == df.height
    _ensure_new_columns(df, dyn_out)

    to_date = ToDateSnapshotAggregator(
        time_column="ts",
        period="90d",
        groupby=["target_str"],
        value_columns=["alcohol"],
        aggregations=["sum", "mean"],
        include_current=True,
        drop_original=False,
    )
    to_date_out = to_date.fit_transform(df)
    assert to_date_out.height == df.height
    _ensure_new_columns(df, to_date_out)

    ewm = EWMAggregator(
        time_column="ts",
        alpha=0.4,
        groupby=["target_str"],
        value_columns=["alcohol"],
        drop_original=False,
    )
    ewm_out = ewm.fit_transform(df)
    assert ewm_out.height == df.height
    _ensure_new_columns(df, ewm_out)


def test_imputers_and_scalers_wine(wine_df: pl.DataFrame) -> None:
    base = wine_df.with_row_index("row_idx").select(
        ["target_str", "alcohol", "color_intensity", "row_idx"]
    )
    df = base.with_columns(
        pl.when((pl.col("row_idx") % 7) == 0).then(None).otherwise(pl.col("alcohol")).alias("alcohol"),
        pl.when((pl.col("row_idx") % 11) == 0)
        .then(None)
        .otherwise(pl.col("color_intensity"))
        .alias("color_intensity"),
    ).drop("row_idx")

    simple = SimpleImputer(
        columns=["alcohol", "target_str"],
        numerical_strategy="mean",
        categorical_strategy="most_frequent",
        add_indicator=True,
        drop_original=False,
    )
    simple_out = simple.fit_transform(df)
    assert simple_out.height == df.height
    _ensure_new_columns(df, simple_out)

    group = GroupImputer(
        groupby=["target_str"],
        columns=["alcohol"],
        numeric_strategy="mean",
        add_indicator=True,
        drop_original=False,
    )
    group_out = group.fit_transform(df)
    assert group_out.height == df.height
    _ensure_new_columns(df, group_out)

    scaler = RobustScaler(columns=["alcohol"], drop_original=False)
    scaled = scaler.fit_transform(wine_df.select(["alcohol"]))
    assert scaled.height == wine_df.height
    _ensure_new_columns(wine_df.select(["alcohol"]), scaled)

    ranker = QuantileRankTransformer(columns=["alcohol"], groupby=["target_str"], drop_original=False)
    rank_out = ranker.fit_transform(wine_df.select(["alcohol", "target_str"]))
    assert rank_out.height == wine_df.height
    _ensure_new_columns(wine_df.select(["alcohol", "target_str"]), rank_out)


def test_outlier_handler_wine(wine_df: pl.DataFrame) -> None:
    df = wine_df.select(["alcohol", "color_intensity"])
    clipper = ClipOutliers(columns=["alcohol"], method="iqr", action="flag")
    flagged = clipper.fit_transform(df)
    assert flagged.height == df.height
    assert "alcohol__outlier" in flagged.columns


def test_auto_featurizer_wine(wine_w_target: pl.DataFrame) -> None:
    df = wine_w_target

    auto = AutoFeaturizer(
        max_ohe_cardinality=5,
        add_ranks=True,
        include_imputation=True,
        include_outliers=True,
        include_vst=True,
        include_discretization=True,
        discretize_bins=4,
        skew_threshold=0.5,
        time_column="ts",
        time_windows=("30d",),
        groupby=["target_str"],
        interaction_aggs=("mean",),
        target_column="target_bin",
        include_target_encoders=True,
    )

    fitted = auto.fit(df)
    plan = fitted.get_plan()

    assert "target" in plan
    # Verify supervised encoders were picked for categorical columns
    target_plan = plan["target"]
    assert any(target_plan.values()), "Expected at least one supervised encoder in plan"

    transformed = auto.transform(df)
    assert transformed.height == df.height
    # Ensure new columns created
    assert set(transformed.columns) != set(df.columns)


def test_suggest_methods_wine(wine_w_target: pl.DataFrame) -> None:
    df = wine_w_target
    plan = suggest_methods(
        df,
        max_ohe_cardinality=5,
        include_imputation=True,
        include_outliers=True,
        include_vst=True,
        include_discretization=True,
        discretize_bins=4,
        skew_threshold=0.5,
        time_column="ts",
        groupby=["target_str"],
        target_column="target_bin",
        include_target_encoders=True,
    )

    assert "categorical" in plan
    assert "numeric" in plan
    assert "target" in plan
    assert any(plan["target"].values()), "Expected supervised recommendations"
