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
# # Stabilizing the UCI Wine Quality Dataset
#
# The red wine quality dataset features continuous chemistry measurements with highly skewed
# distributions. We'll use variance-stabilizing transforms and ranking to tame heavy tails before
# training a regression model on quality scores.
#
# - **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
# - **Task**: Predict sensory quality scores (`quality`)
# - **Features**: 11 physicochemical attributes of red vinho verde samples

# %% [markdown]
# ## Setup

# %%
import polars as pl
from feataz import (
    AutoFeaturizer,
    LogCPTransformer,
    QuantileRankTransformer,
    YeoJohnsonTransformer,
    suggest_methods,
)

pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(12)

# %% [markdown]
# ## Load the dataset

# %%
SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
raw_df = pl.read_csv(SOURCE_URL, separator=";")
raw_df.head()

target_column = "quality"
skewed_columns = ["residual sugar", "chlorides", "total sulfur dioxide", "sulphates"]

# %% [markdown]
# ## Log transform long-tailed features
#
# Adding a constant prior to log transformation (`LogCPTransformer`) keeps zero values safe while
# shrinking long right tails.

# %%
log_transformer = LogCPTransformer(columns=["residual sugar", "chlorides"], c=1.0)
df_log = log_transformer.fit_transform(raw_df)
df_log.select(["residual sugar", "residual sugar__logcp", "chlorides", "chlorides__logcp"]).head()

# %% [markdown]
# ## Symmetric power transforms for near-Gaussian features
#
# `YeoJohnsonTransformer` works for positive and non-positive values. We apply it to sulfur dioxide
# and sulphate readings to reduce skewness without throwing away sign information.

# %%
yeo = YeoJohnsonTransformer(columns=["total sulfur dioxide", "sulphates"])
df_yeo = yeo.fit_transform(df_log)
df_yeo.select(["total sulfur dioxide", "total sulfur dioxide__yeojohnson", "sulphates", "sulphates__yeojohnson"]).head()

# %% [markdown]
# ## Rank encode alcohol strength within quality bands
#
# Ranking alcohol by the target can be useful for tree-based models. `QuantileRankTransformer`
# computes normalized ranks, optionally within groups. Here we rank by the discrete quality score.

# %%
ranker = QuantileRankTransformer(columns=["alcohol"], groupby=[target_column])
df_ranked = ranker.fit_transform(df_yeo)
df_ranked.select([target_column, "alcohol", "alcohol__rank"]).head()

# %% [markdown]
# ## Let feataz suggest complementary transforms
#
# `suggest_methods` looks at the schema to propose a baseline plan that balances
# encoding, scaling, and optional variance stabilizers.

# %%
suggested_plan = suggest_methods(
    raw_df,
    target_column=target_column,
    include_outliers=True,
)
suggested_plan

# %% [markdown]
# ## Run AutoFeaturizer for a turnkey table
#
# The auto-featurizer applies those strategies in one step, producing an
# enriched feature matrix you can compare against the custom recipe.

# %%
auto_featurizer = AutoFeaturizer(
    target_column=target_column,
    include_outliers=True,
    add_ranks=True,
    drop_original=False,
)
auto_df = auto_featurizer.fit_transform(raw_df)
auto_preview_cols = [target_column] + [c for c in auto_df.columns if c != target_column][:9]
auto_df.select(auto_preview_cols).head()

# %% [markdown]
# The engineered view now includes log-compressed sugars and chlorides, symmetric sulfur features,
# and an alcohol rank that respects quality levels. Export the table or plug it into a regression
# pipeline with cross-validation, and compare it with the AutoFeaturizer output for a turnkey
# baseline.
