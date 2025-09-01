feataz — Feature engineering on Polars
======================================

Lightweight feature engineering utilities on top of Polars DataFrames. Provides
categorical encoders, numerical discretizers, variance-stabilizing transforms,
and simple feature interactions with a consistent fit/transform API.

Install
-------

This repo is source-only. Add `src` to your `PYTHONPATH`, e.g.:

    export PYTHONPATH="$PWD/src:$PYTHONPATH"

Optional dependencies (only for tree-based transformers):

- DecisionTreeEncoder, DecisionTreeDiscretizer: `pip install scikit-learn`

Quick start
-----------

```python
import polars as pl
from feataz import (
    OneHotEncoder, OrdinalEncoder, CountFrequencyEncoder, MeanEncoder,
    WeightOfEvidenceEncoder, DecisionTreeEncoder, RareLabelEncoder, StringSimilarityEncoder,
    EqualFrequencyDiscretizer, EqualWidthDiscretizer, ArbitraryDiscretizer, DecisionTreeDiscretizer, GeometricWidthDiscretizer,
    LogTransformer, LogCPTransformer, ReciprocalTransformer, ArcsinTransformer, PowerTransformer, BoxCoxTransformer, YeoJohnsonTransformer,
    FeatureInteractions, TimeSnapshotAggregator,
)

df = pl.DataFrame({
    "cat": ["a", "b", "a", "c", "b", "a"],
    "x": [1.0, 2.5, 0.5, 3.2, 2.7, 4.1],
    "y": [0, 1, 0, 1, 1, 0],
})

# Categorical encoders
df1 = OneHotEncoder(["cat"]).fit_transform(df)
df2 = OrdinalEncoder(["cat"], drop_original=False).fit_transform(df)
df3 = CountFrequencyEncoder(["cat"], normalize=True).fit_transform(df)
df4 = MeanEncoder(target="y", columns=["cat"], smoothing=5).fit_transform(df)
df5 = WeightOfEvidenceEncoder(target="y", columns=["cat"]).fit_transform(df)
df6 = RareLabelEncoder(["cat"], min_frequency=0.3, drop_original=False).fit_transform(df)
df7 = StringSimilarityEncoder(["cat"], top_k_anchors=2).fit_transform(df)

# Discretization
df8 = EqualFrequencyDiscretizer(["x"], n_bins=3).fit_transform(df)
df9 = EqualWidthDiscretizer(["x"], n_bins=4).fit_transform(df)
df10 = GeometricWidthDiscretizer(["x"], n_bins=3).fit_transform(df)
df11 = ArbitraryDiscretizer(bins=[0, 2, 4, float("inf")], columns=["x"]).fit_transform(df)

# Variance-stabilizing transforms
df12 = LogTransformer(["x"]).fit_transform(df)
df13 = LogCPTransformer(["x"], c=1.0).fit_transform(df)
df14 = ReciprocalTransformer(["x"]).fit_transform(df)
df15 = PowerTransformer(power=0.5, columns=["x"]).fit_transform(df)
df16 = BoxCoxTransformer(["x"]).fit_transform(df)
df17 = YeoJohnsonTransformer(["x"]).fit_transform(df)

# Interactions (group-based aggregations)
df18 = FeatureInteractions(groupby=["cat"], value_columns=["x"], aggregations=["sum", "mean"]).fit_transform(df)

# Time snapshots (trailing windows)
df_ts = pl.DataFrame({
    "cat": ["a","a","a","b","b"],
    "ts": pl.date_range(low=pl.datetime(2024,1,1), high=pl.datetime(2024,1,5), interval="1d", eager=True),
    "x": [1.0, 2.0, 3.0, 1.5, 2.5],
})
snap = TimeSnapshotAggregator(
    time_column="ts",
    groupby=["cat"],
    value_columns=["x"],
    windows=["3d", "1w"],
    aggregations=["sum", "mean"],
    include_current=False,
)
df19 = snap.fit_transform(df_ts)
```

Notes
-----

- Tree-based encoders/discretizers require scikit-learn and operate per-column using a univariate tree.
- Box-Cox and Yeo-Johnson lambdas are estimated via a lightweight grid search, no SciPy required.
- String similarity uses Python’s `difflib.SequenceMatcher`; provide anchors or it picks top-k frequent values per column.
- Transformations default to returning new columns with suffixes and typically drop originals; configure via constructor args.
