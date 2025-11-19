# Examples & Tutorials

A few curated resources live inside the repository so you can copy/paste snippets or run full
notebooks without writing everything from scratch.

## Smoke test script

`examples/smoke.py` is a runnable script that exercises almost every transformer shipped in
`src/feataz`. It covers categorical encoders, discretizers, variance stabilizing transforms,
feature builders, imputers, outlier handling, scaling, selection, and the snapshot aggregators.

```bash
python examples/smoke.py
```

Key ideas demonstrated inside the script:

- Reusable column naming conventions: most transformers append a suffix like `__mean` or
  `__qbin`; scanning the printed DataFrames gives you a sense of the resulting schema.
- Graceful degradation when optional dependencies are not installed. Functions that rely on
  scikit-learn are wrapped in `try/except` with clear messaging so you can selectively install
  extras.
- Rolling/snapshot aggregations using `TimeSnapshotAggregator`, `DynamicSnapshotAggregator`,
  `ToDateSnapshotAggregator`, and `EWMAggregator` on a synthetic time series.

Invoke the script after local changes to make sure you did not break the public API, or keep it as a
template when wiring the transformers into a larger Polars pipeline.

## Notebook case study

The `notebook/feature_engineering_on_titanic_data.ipynb` notebook (and its Jupytext version in
`notebook/tutorials/tutorial_01_titanic_survival.py`) walks through a complete binary
classification problem on the Titanic dataset. Highlights:

- Uses `SimpleImputer`, `OneHotEncoder`, `MeanEncoder`, and `ClipOutliers` to produce a tidy table.
- Demonstrates how to call `suggest_methods` for an automatic plan and how `AutoFeaturizer` executes
  that plan end-to-end.
- Shows how to pull CSVs directly from GitHub via Polars streaming APIs.

To run the notebook locally:

```bash
pip install feataz[sklearn] jupyter jupytext
jupyter lab notebook/feature_engineering_on_titanic_data.ipynb
```

(Use `jupytext --sync notebook/tutorials/tutorial_01_titanic_survival.py` if you want to edit the
Python version and keep the `.ipynb` file in sync.)

## Tutorial gallery

The `notebook/tutorials/` directory contains ten lightweight notebooks targeting different public
datasets (iris, wine quality, California housing, bike sharing time series, etc.). Each notebook:

1. Loads the dataset with Polars.
2. Applies a curated subset of `feataz` transformers.
3. Prints intermediate DataFrames so you can see how columns evolve after every step.

Launch the gallery with your favorite notebook runner:

```bash
jupyter lab notebook/tutorials/tutorial_05_heart_disease.py  # automatically converted via jupytext
```

These notebooks double as regression tests—when you change a transformer, rerun a few tutorials to
see whether column names, row counts, or descriptive statistics drifted unexpectedly.
