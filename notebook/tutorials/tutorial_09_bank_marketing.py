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
# # Marketing Response Modeling with feataz
#
# The Portuguese bank marketing dataset contains categorical campaign metadata and continuous call
# statistics. We'll prepare the target, collapse infrequent values, compute contact frequencies, and
# mean-encode socio-demographics.
#
# - **Source**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
#   via [Jason Brownlee's mirror](https://raw.githubusercontent.com/jbrownlee/Datasets/master/bank_marketing.csv)
# - **Task**: Predict whether the client subscribed (`y`)
# - **Features**: Demographics, economic indicators, contact outcomes

# %% [markdown]
# ## Setup

# %%
import polars as pl
from feataz import (
    AutoFeaturizer,
    CountFrequencyEncoder,
    MeanEncoder,
    RareLabelEncoder,
    suggest_methods,
)

pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(14)

# %% [markdown]
# ## Load and prepare the dataset

# %%
SOURCE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/bank_marketing.csv"
raw_df = pl.read_csv(SOURCE_URL)
raw_df = raw_df.rename({"y": "subscribed"})
raw_df.head()

target_column = "subscribed"
categorical_columns = ["job", "marital", "education", "contact", "month", "poutcome"]

# %% [markdown]
# ## Binary target column
#
# Convert the `yes`/`no` target into a numeric indicator for use with target encoders and logistic
# regression.

# %%
df_target = raw_df.with_columns(
    pl.when(pl.col(target_column) == "yes").then(1).otherwise(0).alias("subscribed__int")
)
df_target.select([target_column, "subscribed__int"]).head()

# %% [markdown]
# ## Consolidate rare campaign outcomes
#
# Rare `poutcome` and job categories get mapped into an "other" bucket, while flags mark affected
# rows.

# %%
rare_encoder = RareLabelEncoder(columns=["job", "poutcome"], min_frequency=0.02, replace_with="other", drop_original=False)
df_rare = rare_encoder.fit_transform(df_target)
df_rare.select(["job", "poutcome", "job__is_rare", "poutcome__is_rare"]).head()

# %% [markdown]
# ## Contact frequency encoding
#
# Frequency encoding of contact channel and month reveals how common certain strategies are in the
# dataset.

# %%
freq_encoder = CountFrequencyEncoder(columns=["contact", "month"], normalize=True)
df_freq = freq_encoder.fit_transform(df_rare)
df_freq.select(["contact", "contact__countfreq", "month", "month__countfreq"]).head()

# %% [markdown]
# ## Smoothed target means for demographics
#
# Mean encoding job and marital status with smoothing captures subtle response rate differences while
# shrinking toward the global average for small groups.

# %%
mean_encoder = MeanEncoder(target="subscribed__int", columns=["job", "marital"], smoothing=15.0)
df_mean = mean_encoder.fit_transform(df_freq)
df_mean.select(["job", "job__mean", "marital", "marital__mean", "subscribed__int"]).head()

# %% [markdown]
# ## Suggested baseline plan for campaign data
#
# Feed the prepared frame with the numeric target into `suggest_methods` to see
# what feataz would automate for you.

# %%
auto_target_column = "subscribed__int"
auto_base_df = df_target.drop(target_column)
suggested_plan = suggest_methods(
    auto_base_df,
    target_column=auto_target_column,
    include_outliers=True,
)
suggested_plan

# %% [markdown]
# ## AutoFeaturizer preview
#
# AutoFeaturizer can execute the recommended transformations and produce an
# augmented modeling table alongside the engineered features above.

# %%
auto_featurizer = AutoFeaturizer(
    target_column=auto_target_column,
    include_outliers=True,
    add_ranks=True,
    drop_original=False,
)
auto_df = auto_featurizer.fit_transform(auto_base_df)
auto_preview_cols = [auto_target_column] + [c for c in auto_df.columns if c != auto_target_column][:9]
auto_df.select(auto_preview_cols).head()

# %% [markdown]
# Combine encoded features with duration/campaign counts and macro indicators (`pdays`, `previous`)
# when building the final model, and validate against the AutoFeaturizer plan for a quick baseline.
