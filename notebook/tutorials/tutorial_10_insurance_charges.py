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
# # Health Insurance Charges Feature Engineering
#
# The insurance charges dataset blends categorical risk factors with skewed numeric targets. We'll
# binary-encode the smoking flag, expand the region column, log-transform the target, and create
# relative indicators to BMI.
#
# - **Source**: [stedy/Machine-Learning-with-R-datasets](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv)
# - **Task**: Predict medical costs (`charges`)
# - **Features**: Age, sex, BMI, dependent count, smoking status, region

# %% [markdown]
# ## Setup

# %%
import polars as pl
from feataz import (
    AutoFeaturizer,
    BinaryEncoder,
    LogTransformer,
    OneHotEncoder,
    RelativeFeatures,
    suggest_methods,
)

pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(12)

# %% [markdown]
# ## Load the dataset

# %%
SOURCE_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
raw_df = pl.read_csv(SOURCE_URL)
raw_df.head()

target_column = "charges"

# %% [markdown]
# ## Binary encode smoking status
#
# `BinaryEncoder` splits a categorical feature into binary digits, which is useful for high-variance
# two-level columns when we want a compact representation.

# %%
smoker_encoder = BinaryEncoder(columns=["smoker"])
df_smoker = smoker_encoder.fit_transform(raw_df)
df_smoker.select([c for c in df_smoker.columns if c.startswith("smoker__binary")]).head()

# %% [markdown]
# ## One-hot encode the geographic region
#
# Regional differences in healthcare costs can be captured with one-hot indicators.

# %%
region_encoder = OneHotEncoder(columns=["region"], drop_original=False)
df_region = region_encoder.fit_transform(df_smoker)
df_region.select([c for c in df_region.columns if c.startswith("region_")]).head()

# %% [markdown]
# ## Log-transform the skewed cost target
#
# Medical cost data are highly skewed. Applying a log transform keeps the scale manageable for linear
# models. We preserve the original target alongside the transformed variant.

# %%
log_transformer = LogTransformer(columns=[target_column])
df_log = log_transformer.fit_transform(df_region)
df_log.select([target_column, f"{target_column}__log"]).head()

# %% [markdown]
# ## Relative features vs. BMI
#
# Ratios and percentage differences of charges relative to BMI help highlight cost outliers after
# controlling for body mass.

# %%
relative = RelativeFeatures(
    target_columns=[target_column],
    reference_columns=["bmi"],
    operations=["ratio", "pct_diff"],
)
df_relative = relative.fit_transform(df_log)
df_relative.select(["bmi", target_column, f"{target_column}__ratio_bmi", f"{target_column}__pct_diff_bmi"]).head()

# %% [markdown]
# ## Quick suggestions for further automation
#
# `suggest_methods` surfaces additional feataz transformers that align with the
# schema and regression target.

# %%
suggested_plan = suggest_methods(
    raw_df,
    target_column=target_column,
    include_outliers=True,
)
suggested_plan

# %% [markdown]
# ## AutoFeaturizer baseline features
#
# Let AutoFeaturizer execute the automatic plan and preview the resulting table
# before layering the custom engineering above.

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
# The final table combines binary-encoded smoke status, regional one-hots, log costs, and BMI-relative
# indicators. Pair these with age and dependents for a full regression design matrix, and keep the
# AutoFeaturizer output nearby for automated experimentation.
