# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Iris Flower Feature Recipes with feataz
#
# The Fisher iris dataset is a gentle playground for experimenting with numerical feature
# transformations. We'll bucketize petal measurements, derive polynomial-style interactions,
# and create relative ratios to capture shape differences across species.
#
# - **Source**: [uiuc-cse/data-fa14](https://github.com/uiuc-cse/data-fa14/blob/gh-pages/data/iris.csv)
# - **Task**: Multi-class classification (`species`)
# - **Features**: Sepal and petal lengths/widths in centimeters

# %% [markdown]
# ## Setup

# %%
import polars as pl
from feataz import (
    AutoFeaturizer,
    EqualWidthDiscretizer,
    MathFeatures,
    RelativeFeatures,
    suggest_methods,
)

pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(12)

# %% [markdown]
# ## Load the dataset

# %%
SOURCE_URL = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
raw_df = pl.read_csv(SOURCE_URL)
raw_df.head()

feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target_column = "species"

# %% [markdown]
# ## Discretize continuous measurements
#
# `EqualWidthDiscretizer` creates ordinal bins that can be useful for tree ensembles or when you
# want to later apply target encoders on coarse categories.

# %%
discretizer = EqualWidthDiscretizer(columns=feature_columns, n_bins=4)
df_binned = discretizer.fit_transform(raw_df)
df_binned.select([c for c in df_binned.columns if c.endswith("__bin")] + feature_columns).head()

# %% [markdown]
# ## Polynomial-style math combinations
#
# `MathFeatures` can generate unary transforms and pairwise interactions in a single call. Here we
# build square roots, squares, and pairwise sums/differences across the petal measurements.

# %%
math_features = MathFeatures(
    columns=feature_columns,
    unary_ops=["sqrt"],
    powers=[2],
    binary_ops=("add", "sub"),
)
df_math = math_features.fit_transform(df_binned)
df_math.select([c for c in df_math.columns if c.endswith("__sqrt") or c.endswith("__pow_2") or "__add" in c][:8]).head()

# %% [markdown]
# ## Relative ratios highlight shape differences
#
# Ratios and percentage differences between petals and sepals can capture the structural variety
# that distinguishes iris species.

# %%
relative = RelativeFeatures(
    target_columns=["petal_length", "petal_width"],
    reference_columns=["sepal_length", "sepal_width"],
    operations=["ratio", "diff", "pct_diff"],
)
df_relative = relative.fit_transform(df_math)
df_relative.select([c for c in df_relative.columns if "petal" in c and ("__ratio" in c or "__pct_diff" in c or "__diff" in c)]).head()

# %% [markdown]
# ## Suggested automated feature plan
#
# Before diving deeper, have `feataz` outline which default transformers would
# pair well with the dataset.

# %%
suggested_plan = suggest_methods(raw_df, target_column=target_column)
suggested_plan

# %% [markdown]
# ## Quick AutoFeaturizer baseline
#
# `AutoFeaturizer` turns the suggestions into a ready-made feature matrix that
# you can feed to a model or use as a comparison point for bespoke recipes.

# %%
auto_featurizer = AutoFeaturizer(
    target_column=target_column,
    add_ranks=True,
    drop_original=False,
)
auto_df = auto_featurizer.fit_transform(raw_df)
auto_preview_cols = [target_column] + [c for c in auto_df.columns if c != target_column][:9]
auto_df.select(auto_preview_cols).head()

# %% [markdown]
# With discretized bins, polynomial expansions, and relative ratios, we can feed a classifier a
# richer representation of each flower. Try experimenting with class-specific aggregations or
# combining the engineered features with cross-validated target encoders when training a final
# estimator, and benchmark them against the automatically generated AutoFeaturizer baseline.
