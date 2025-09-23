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
# # Encoding the Mushroom Edibility Dataset
#
# Mushroom edibility prediction is a purely categorical problem with compact alphabets. We'll use a
# combination of rare label grouping, ordinal encodings, and hashing to create dense numerical
# features that scale to wide one-hot expansions.
#
# - **Source**: [EPFL ML course datasets](https://github.com/epfml/ML_course/blob/master/labs/ex11/mushrooms.csv)
# - **Task**: Predict whether a mushroom is edible (`class`)
# - **Features**: Cap shape/size/color, gill descriptors, ring types, odors, etc.

# %% [markdown]
# ## Setup

# %%
import polars as pl
from feataz import (
    AutoFeaturizer,
    HashEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
    suggest_methods,
)

pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(20)

# %% [markdown]
# ## Load the dataset

# %%
SOURCE_URL = "https://raw.githubusercontent.com/epfml/ML_course/master/labs/ex11/mushrooms.csv"
raw_df = pl.read_csv(SOURCE_URL)
raw_df.head()

target_column = "class"
feature_columns = [col for col in raw_df.columns if col != target_column]

# %% [markdown]
# ## Group rare categories before encoding
#
# Even small rare-category probabilities can introduce high-variance encodings. We'll gather any
# category that accounts for fewer than 1% of observations into a shared "rare" bucket while keeping
# flags about which columns were affected.

# %%
rare_encoder = RareLabelEncoder(columns=feature_columns, min_frequency=0.01, replace_with="rare", drop_original=False)
df_rare = rare_encoder.fit_transform(raw_df)
df_rare.select([c for c in df_rare.columns if c.endswith("__is_rare")][:10]).head()

# %% [markdown]
# ## Ordinal encoding for dense numeric tables
#
# Ordinal encoders map each category to an integer code. This is a quick way to build numeric tables
# for tree models or to feed subsequent hashing encoders.

# %%
ordinal_encoder = OrdinalEncoder(columns=feature_columns, drop_original=False)
df_ordinal = ordinal_encoder.fit_transform(df_rare)
df_ordinal.select([c for c in df_ordinal.columns if c.endswith("__ordinal")][:10]).head()

# %% [markdown]
# ## Hash trick for wide categoricals
#
# To capture cross-feature interactions without exploding dimensionality, we can hash a few
# informative columns (odor and spore print color) into a fixed-size numeric embedding.

# %%
hash_encoder = HashEncoder(columns=["odor", "spore-print-color"], n_components=16)
df_hashed = hash_encoder.fit_transform(df_ordinal)
df_hashed.select([c for c in df_hashed.columns if c.startswith("odor__hash") or c.startswith("spore-print-color__hash")][:8]).head()

# %% [markdown]
# ## Peek at automated recommendations
#
# Even though the dataset is all-categorical, `suggest_methods` will highlight
# the encoders AutoFeaturizer would choose by default.

# %%
suggested_plan = suggest_methods(raw_df, target_column=target_column)
suggested_plan

# %% [markdown]
# ## AutoFeaturizer in action
#
# Run the auto featurizer to compare its default encodings with the custom
# pipeline above. Keeping the originals makes it easy to blend strategies.

# %%
auto_featurizer = AutoFeaturizer(
    target_column=target_column,
    drop_original=False,
)
auto_df = auto_featurizer.fit_transform(raw_df)
auto_preview_cols = [target_column] + [c for c in auto_df.columns if c != target_column][:9]
auto_df.select(auto_preview_cols).head()

# %% [markdown]
# Combine ordinal, rare label, and hashed features to construct a high-signal classifier for mushroom
# edibility. For linear models, you can additionally apply one-hot encoding on the hashed outputs for
# sparsity-friendly representations, and keep the AutoFeaturizer result handy as a quick baseline.
