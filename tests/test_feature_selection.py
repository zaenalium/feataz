import numpy as np
import polars as pl

from feataz.selection import (
    VarianceThresholdSelector,
    MutualInformationSelector,
    ModelBasedImportanceSelector,
    MRMRSelector,
)


def test_variance_threshold_selector_drops_low_variance_columns():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "constant": [5.0, 5.0, 5.0, 5.0],
            "varying": [0.0, 1.0, 2.0, 3.0],
            "low_var": [1.0, 1.0, 1.1, 0.9],
        }
    )

    selector = VarianceThresholdSelector(threshold=0.01)
    result = selector.fit(df).transform(df)

    assert "constant" not in result.columns
    assert "low_var" not in result.columns
    assert "varying" in result.columns
    assert selector.dropped_features_ == ["constant", "low_var"]
    assert selector.get_support() == [True, False, True, False]
    assert selector.variances_["constant"] == 0.0
    assert selector.variances_["low_var"] < 0.01


def test_variance_threshold_selector_respects_column_subset():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "constant": [5.0, 5.0, 5.0, 5.0],
            "varying": [0.0, 1.0, 2.0, 3.0],
            "low_var": [1.0, 1.0, 1.1, 0.9],
        }
    )

    selector = VarianceThresholdSelector(threshold=0.01, columns=["varying", "low_var"])
    selector.fit(df)
    result = selector.transform(df)

    assert selector.feature_names_in_ == ["varying", "low_var"]
    assert selector.get_selected_features() == ["varying"]
    assert "constant" in result.columns
    assert "low_var" not in result.columns
    assert "varying" in result.columns


def test_mutual_information_selector_prefers_informative_feature():
    df = pl.DataFrame(
        {
            "x1": [0, 1, 0, 1, 0, 1, 0, 1],
            "x2": [0, 0, 1, 1, 0, 0, 1, 1],
            "noise": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    df = df.with_columns((pl.col("x1") | pl.col("x2")).cast(pl.Int8).alias("target"))

    selector = MutualInformationSelector(target="target", k=2, threshold=0.0)
    result = selector.fit(df).transform(df)

    assert set(selector.get_selected_features()) == {"x1", "x2"}
    assert "noise" not in result.columns
    assert selector.problem_ == "classification"


def test_model_based_importance_selector_uses_default_estimator():
    df = pl.DataFrame(
        {
            "x1": [0, 1, 0, 1, 0, 1, 0, 1],
            "x2": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    df = df.with_columns((pl.col("x1") ^ pl.col("x2")).cast(pl.Int8).alias("target"))

    selector = ModelBasedImportanceSelector(target="target", k=1)
    result = selector.fit(df).transform(df)

    assert selector.estimator_ is not None
    assert selector.get_selected_features() == ["x1"] or selector.get_selected_features() == ["x2"]
    assert "target" in result.columns
    assert len(selector.get_support()) == 2


def test_mrmr_selector_penalizes_redundant_features():
    rng = np.random.default_rng(0)
    n = 48
    x1 = rng.integers(0, 2, size=n)
    x2 = x1.copy()
    x3 = rng.integers(0, 2, size=n)
    target = ((x1 + x3) > 0).astype(int)

    df = pl.DataFrame({
        "x1": x1.tolist(),
        "x2": x2.tolist(),
        "x3": x3.tolist(),
        "target": target.tolist(),
    })

    selector = MRMRSelector(target="target", k=2)
    selector.fit(df)
    selected = selector.get_selected_features()

    assert len(selected) == 2
    assert "x3" in selected
    assert set(selected) & {"x1", "x2"}
