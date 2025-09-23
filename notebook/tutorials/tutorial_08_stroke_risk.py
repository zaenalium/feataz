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
# # Stroke Risk Feature Engineering
#
# Healthcare risk scoring combines demographic categoricals with biometric measurements. We'll handle
# missing BMI values in a group-aware way, encode lifestyle variables, and rank glucose levels by
# gender.
#
# - **Source**: [Stroke Prediction Dataset](https://github.com/Harshita-Kanal/Stroke-Prediction-Dataset/blob/master/healthcare-dataset-stroke-data.csv)
# - **Task**: Predict stroke occurrence (`stroke`)
# - **Features**: Age, hypertension, cardiovascular history, lifestyle, glucose, BMI

# %% [markdown]
# ## Setup

# %%
import polars as pl
from feataz import (
    AutoFeaturizer,
    GroupImputer,
    OneHotEncoder,
    QuantileRankTransformer,
    suggest_methods,
)

pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(12)

# %% [markdown]
# ## Load the dataset

# %%
SOURCE_URL = "https://raw.githubusercontent.com/Harshita-Kanal/Stroke-Prediction-Dataset/master/healthcare-dataset-stroke-data.csv"
raw_df = pl.read_csv(SOURCE_URL)
raw_df.head()

target_column = "stroke"
numeric_columns = ["age", "avg_glucose_level", "bmi"]
lifestyle_columns = ["ever_married", "work_type", "Residence_type", "smoking_status"]

# %% [markdown]
# ## Group-based BMI imputation
#
# Body mass index is missing for a subset of patients. Filling the gaps with a population-wide mean
# may bias results, so we impute within gender groups.

# %%
bmi_imputer = GroupImputer(groupby=["gender"], columns=["bmi"], numeric_strategy="mean", add_indicator=True)
df_imputed = bmi_imputer.fit_transform(raw_df)
df_imputed.select(["gender", "bmi", "bmi__imputed", "bmi__was_missing"]).head()

# %% [markdown]
# ## One-hot encode lifestyle attributes
#
# Lifestyle categories such as work type, residence, and smoking status are low-cardinality and are
# ideal for one-hot encoding.

# %%
ohe = OneHotEncoder(columns=lifestyle_columns, drop_original=False)
df_encoded = ohe.fit_transform(df_imputed)
df_encoded.select([c for c in df_encoded.columns if any(col in c for col in lifestyle_columns)][:12]).head()

# %% [markdown]
# ## Rank glucose by gender
#
# Ranking glucose within gender highlights relative risk in each demographic group.

# %%
ranker = QuantileRankTransformer(columns=["avg_glucose_level"], groupby=["gender"])
df_ranked = ranker.fit_transform(df_encoded)
df_ranked.select(["gender", "avg_glucose_level", "avg_glucose_level__rank"]).head()

# %% [markdown]
# ## Automated suggestions for this mix of features
#
# `suggest_methods` captures the default feataz recommendations for imputation,
# encoding, and variance handling given the stroke dataset.

# %%
suggested_plan = suggest_methods(
    raw_df,
    target_column=target_column,
    include_outliers=True,
)
suggested_plan

# %% [markdown]
# ## AutoFeaturizer to compare against manual steps
#
# The auto pipeline can serve as a baseline table before layering the custom
# group-aware imputations and encodings demonstrated above.

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
# The engineered frame now includes BMI imputations with indicators, encoded lifestyle variables, and
# normalized glucose rankings. These features can feed linear models or boosted trees with minimal
# additional preprocessing, and the AutoFeaturizer table provides a solid automated baseline.
