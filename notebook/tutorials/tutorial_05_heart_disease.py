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
# # Preparing the Cleveland Heart Disease Dataset
#
# The heart disease dataset mixes categorical stress-test outcomes with continuous biometrics. We'll
# consolidate rare categories, build weight-of-evidence encodings, and robust-scale skewed vitals.
#
# - **Source**: [plotly/datasets](https://github.com/plotly/datasets/blob/master/heart.csv)
# - **Task**: Predict the presence of heart disease (`target`)
# - **Features**: Demographics, resting measurements, and exercise test results

# %% [markdown]
# ## Setup

# %%
import polars as pl
from feataz import (
    AutoFeaturizer,
    RareLabelEncoder,
    RobustScaler,
    WeightOfEvidenceEncoder,
    suggest_methods,
)

pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(12)

# %% [markdown]
# ## Load the dataset

# %%
SOURCE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
raw_df = pl.read_csv(SOURCE_URL)
raw_df.head()

target_column = "target"
categorical_columns = ["cp", "restecg", "slope", "thal"]
numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# %% [markdown]
# ## Collapse rare stress-test outcomes
#
# Some categorical levels appear only a handful of times. `RareLabelEncoder` groups them into an
# "other" bucket so that encoders and models are less sensitive to sampling noise.

# %%
rare_encoder = RareLabelEncoder(columns=categorical_columns, min_frequency=0.05, replace_with="other", drop_original=False)
df_rare = rare_encoder.fit_transform(raw_df)
df_rare.select(categorical_columns + [f"{col}__is_rare" for col in categorical_columns]).head()

# %% [markdown]
# ## Weight of evidence encodings
#
# Weight-of-evidence encoders map categorical levels to log odds of the positive class. They work
# well for logistic regression or as informative features for boosted trees.

# %%
woe_encoder = WeightOfEvidenceEncoder(target=target_column, columns=categorical_columns)
df_woe = woe_encoder.fit_transform(df_rare)
df_woe.select([target_column] + [c for c in df_woe.columns if c.endswith("__woe")]).head()

# %% [markdown]
# ## Robust scaling for vitals
#
# Median and IQR scaling dampens the impact of heavy-tailed biometrics (like cholesterol) and keeps
# them on comparable ranges.

# %%
scaler = RobustScaler(columns=numeric_columns)
df_scaled = scaler.fit_transform(df_woe)
df_scaled.select([f"{col}__robust" for col in numeric_columns]).head()

# %% [markdown]
# ## Automatically suggested encoders and scalers
#
# `suggest_methods` offers a quick overview of which default feataz transformers
# align with the mix of numeric and categorical columns.

# %%
suggested_plan = suggest_methods(
    raw_df,
    target_column=target_column,
    include_outliers=True,
)
suggested_plan

# %% [markdown]
# ## AutoFeaturizer snapshot
#
# Apply the end-to-end AutoFeaturizer to compare its default engineering choices
# with the tailored workflow above.

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
# The engineered dataset now contains WOE-encoded categoricals, rare indicators, and scale-stabilized
# vitals suitable for linear or nonlinear classifiers, with the AutoFeaturizer plan offering a handy
# benchmark.
