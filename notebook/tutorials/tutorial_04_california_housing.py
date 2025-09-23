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
# # Encoding California Housing Tracts
#
# The California housing dataset captures demographic aggregates per census block group. We'll
# create frequency encodings for the categorical `ocean_proximity`, aggregate summary statistics per
# proximity, and derive income ranks to better capture spatial heterogeneity.
#
# - **Source**: [Hands-On ML 2e companion data](https://github.com/ageron/handson-ml2/blob/master/datasets/housing/housing.csv)
# - **Task**: Predict median house values (`median_house_value`)
# - **Features**: Location, age, household counts, and categorical ocean proximity labels

# %% [markdown]
# ## Setup

# %%
import polars as pl
from feataz import (
    AutoFeaturizer,
    CountFrequencyEncoder,
    FeatureInteractions,
    QuantileRankTransformer,
    suggest_methods,
)

pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(12)

# %% [markdown]
# ## Load the dataset

# %%
SOURCE_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
raw_df = pl.read_csv(SOURCE_URL)
raw_df.head()

category_column = "ocean_proximity"
value_columns = ["median_income", "median_house_value"]

# %% [markdown]
# ## Frequency encoding for ocean proximity
#
# Frequency encoders preserve the cardinality information (how common each category is) while keeping
# a single numeric column per feature. We normalize the counts to lie on [0, 1].

# %%
freq_encoder = CountFrequencyEncoder(columns=[category_column], normalize=True)
df_freq = freq_encoder.fit_transform(raw_df)
df_freq.select([category_column, f"{category_column}__countfreq"]).head()

# %% [markdown]
# ## Aggregate neighborhood statistics
#
# Aggregating median income and value per ocean proximity category surfaces regional context. The
# resulting features can help a regression model understand how a block group compares to its peers.

# %%
interactions = FeatureInteractions(
    groupby=[category_column],
    value_columns=value_columns,
    aggregations=["mean", "max"],
)
df_interactions = interactions.fit_transform(df_freq)
df_interactions.select([category_column] + [c for c in df_interactions.columns if c.endswith("__mean") or c.endswith("__max")][:6]).head()

# %% [markdown]
# ## Income rank within proximity bands
#
# Ranking median income relative to other block groups with the same `ocean_proximity` label injects
# a local ordering signal that can be especially informative for tree-based models.

# %%
ranker = QuantileRankTransformer(columns=["median_income"], groupby=[category_column])
df_ranked = ranker.fit_transform(df_interactions)
df_ranked.select([category_column, "median_income", "median_income__rank"]).head()

# %% [markdown]
# ## Use feataz to outline an automatic plan
#
# The helper inspects column types to recommend encoders, scalers, and optional
# group aggregations for later automation.

# %%
suggested_plan = suggest_methods(
    raw_df,
    target_column="median_house_value",
    groupby=[category_column],
    include_outliers=True,
)
suggested_plan

# %% [markdown]
# ## AutoFeaturizer for a full pipeline in one call
#
# Apply the plan end-to-end, keeping original columns so you can blend manual
# features with the automatically generated ones.

# %%
auto_featurizer = AutoFeaturizer(
    target_column="median_house_value",
    groupby=[category_column],
    include_outliers=True,
    add_ranks=True,
    drop_original=False,
)
auto_df = auto_featurizer.fit_transform(raw_df)
auto_preview_cols = ["median_house_value"] + [c for c in auto_df.columns if c != "median_house_value"][:9]
auto_df.select(auto_preview_cols).head()

# %% [markdown]
# Combine these engineered columns with raw numeric totals or ratios such as rooms per household for a
# complete modeling dataset, or jump-start experimentation with the automatically generated
# AutoFeaturizer table.
