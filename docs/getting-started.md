# Getting started

This page covers installation, configuration, and the quickest way to convince yourself that the
transformers in `feataz` work for your Polars pipelines.

## Installation

```bash
pip install feataz
```

Optional extras:

- Decision-tree and scikit-learn powered helpers:

  ```bash
  pip install feataz[sklearn]
  ```

- String similarity acceleration and numerical routines (for larger data sets):

  ```bash
  pip install feataz[accelerated]
  ```

From a local checkout you can install in editable mode:

```bash
pip install -e .
```

## Quick start

```python
import polars as pl
from feataz import (
    OneHotEncoder,
    MeanEncoder,
    TimeSnapshotAggregator,
    MathFeatures,
    SimpleImputer,
    ClipOutliers,
    QuantileRankTransformer,
)

df = pl.DataFrame({
    "cat": ["a", "b", "a", "c", "b", "a"],
    "x": [1.0, 2.5, 0.5, 3.2, 2.7, 4.1],
    "y": [0, 1, 0, 1, 1, 0],
})

ohe = OneHotEncoder(["cat"], drop_original=False)
df_ohe = ohe.fit_transform(df)

mean_enc = MeanEncoder(target="y", columns=["cat"], smoothing=5.0)
df_mean = mean_enc.fit_transform(df_ohe)

snap = TimeSnapshotAggregator(
    time_column="ts",
    groupby=["cat"],
    value_columns=["x"],
    windows=["3d", "1w"],
    aggregations=["sum", "mean"],
)
# .fit/transform on a time series DataFrame

math = MathFeatures(columns=["x"], unary_ops=["log", "sqrt"], powers=[2])
scaler = QuantileRankTransformer(columns=["x"], groupby=["cat"])
imputer = SimpleImputer(
    numeric_strategy_map={"x": "median"},
    add_indicator=True,
)
clipper = ClipOutliers(columns=["x"], method="iqr", action="clip")

result = scaler.fit_transform(
    clipper.fit_transform(
        math.fit_transform(
            imputer.fit_transform(df_mean)
        )
    )
)
print(result)
```

Every transformer stores its learned state in attributes ending with `_`. Use `get_feature_names_out`
to inspect generated column names and `to_dict()` / `from_dict()` for lightweight persistence.

## Suggested workflows

- Start with [`suggest_methods`](reference/index.md#autofeaturizer-and-suggestions) to obtain a
  plan for a previously unseen dataset. Feed the result into `AutoFeaturizer` when you want a
  single object that orchestrates the suggested steps.
- Use the [smoke test example](examples.md#smoke-test-script) to sanity check a local build or to
  observe column naming conventions before you integrate transformers into production pipelines.
- Run the [Titanic tutorial notebook](examples.md#tutorial-gallery) to see how to chain multiple
  modules, export artifacts, and compare automatically engineered columns against manual baselines.
