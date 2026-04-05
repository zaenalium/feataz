"""Tests for LazyFrame support."""
from __future__ import annotations

import polars as pl
import pytest

from feataz import MathFeatures
from feataz import RelativeFeatures
from feataz import CyclicalFeatures
from feataz import OneHotEncoder
from feataz import OrdinalEncoder
from feataz import EqualFrequencyDiscretizer
from feataz import EqualWidthDiscretizer
from feataz import SimpleImputer
from feataz import RobustScaler


class TestLazyFrameMathFeatures:
    def test_lazy_transform(self):
        lf = pl.LazyFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        t = MathFeatures(columns=["a", "b"], unary_ops=["log"], drop_original=False)
        df_train = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        t.fit(df_train)
        result = t.transform(lf)
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "a__log" in collected.columns
        assert "b__log" in collected.columns

    def test_dataframe_still_works(self):
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        t = MathFeatures(columns=["a", "b"], unary_ops=["log"])
        t.fit(df)
        result = t.transform(df)
        assert isinstance(result, pl.DataFrame)
        assert "a__log" in result.columns


class TestLazyFrameEncoders:
    def test_onehot_lazy(self):
        lf = pl.LazyFrame({
            "cat": ["a", "b", "a"],
            "num": [1, 2, 3],
        })
        t = OneHotEncoder(columns=["cat"])
        df_train = pl.DataFrame({"cat": ["a", "b", "c"], "num": [1, 2, 3]})
        t.fit(df_train)
        result = t.transform(lf)
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "cat__a" in collected.columns
        assert "cat__b" in collected.columns

    def test_ordinal_lazy(self):
        lf = pl.LazyFrame({
            "cat": ["a", "b", "a"],
            "num": [1, 2, 3],
        })
        t = OrdinalEncoder(columns=["cat"])
        df_train = pl.DataFrame({"cat": ["a", "b", "c"], "num": [1, 2, 3]})
        t.fit(df_train)
        result = t.transform(lf)
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "cat__ord" in collected.columns


class TestLazyFrameDiscretizers:
    def test_equal_frequency_lazy(self):
        lf = pl.LazyFrame({
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        t = EqualFrequencyDiscretizer(columns=["val"], n_bins=3)
        df_train = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        t.fit(df_train)
        result = t.transform(lf)
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "val__qbin" in collected.columns

    def test_equal_width_lazy(self):
        lf = pl.LazyFrame({
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        t = EqualWidthDiscretizer(columns=["val"], n_bins=3)
        df_train = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        t.fit(df_train)
        result = t.transform(lf)
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "val__wbin" in collected.columns


class TestLazyFrameImputers:
    def test_simple_imputer_lazy(self):
        df_train = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        t = SimpleImputer(columns=["val"], numerical_strategy="mean", drop_original=True)
        t.fit(df_train)
        lf = pl.LazyFrame({
            "val": [1.0, None, 3.0, None, 5.0],
        })
        result = t.transform(lf)
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert collected["val"].null_count() == 0


class TestLazyFrameScalers:
    def test_robust_scaler_lazy(self):
        lf = pl.LazyFrame({
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        t = RobustScaler(columns=["val"])
        df_train = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        t.fit(df_train)
        result = t.transform(lf)
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "val__rsc" in collected.columns


class TestLazyFrameCyclical:
    def test_cyclical_lazy(self):
        lf = pl.LazyFrame({
            "hour": [0, 6, 12, 18, 23],
        })
        t = CyclicalFeatures(columns=["hour"], period=24)
        t.fit(pl.DataFrame({"hour": [0, 6, 12]}))
        result = t.transform(lf)
        assert isinstance(result, pl.LazyFrame)
        collected = result.collect()
        assert "hour__sin" in collected.columns
        assert "hour__cos" in collected.columns
