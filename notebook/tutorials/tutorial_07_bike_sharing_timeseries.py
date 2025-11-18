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
# # Time-Aware Features for the Bike Sharing Dataset
#
# Daily bike rental counts fluctuate with seasonality and weather. We'll engineer cyclical calendar
# features, rolling summaries, and exponential weighted means to capture temporal structure.
#
# - **Source**: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
#   via [Jason Brownlee's mirror](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-bike-share.csv)
# - **Task**: Forecast total rentals (`cnt`)
# - **Features**: Calendar indicators, weather, temperature, registered/casual rider counts

# %% [markdown]
# ## Setup

# %%
import polars as pl
from feataz import (
    AutoFeaturizer,
    CyclicalFeatures,
    EWMAggregator,
    TimeSnapshotAggregator,
    suggest_methods,
)

pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(16)

# %% [markdown]
# ## Load and prepare the dataset
#
# `try_parse_dates=True` automatically converts ISO date strings to `pl.Datetime` objects.

# %%
SOURCE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-bike-share.csv"
raw_df = pl.read_csv(SOURCE_URL, try_parse_dates=True)
raw_df = raw_df.rename({"count": "cnt"}) if "count" in raw_df.columns else raw_df
raw_df.head()

time_column = "dteday"
target_column = "cnt"

# %% [markdown]
# ## Encode cyclical calendar signals
#
# Month and weekday are cyclical. `CyclicalFeatures` projects them to sin/cos pairs to preserve
# continuity.

# %%
month_cyc = CyclicalFeatures(columns=["mnth"], period=12)
df_month = month_cyc.fit_transform(raw_df)

weekday_cyc = CyclicalFeatures(columns=["weekday"], period=7)
df_cyc = weekday_cyc.fit_transform(df_month)
df_cyc.select(["mnth__sin", "mnth__cos", "weekday__sin", "weekday__cos"]).head()

# %% [markdown]
# ## Trailing window summaries by season
#
# `TimeSnapshotAggregator` computes trailing-window stats per group. We'll compute 7- and 30-day
# averages of rentals within each season to capture short- and medium-term momentum.

# %%
snapshot = TimeSnapshotAggregator(
    time_column=time_column,
    groupby=["season"],
    value_columns=[target_column],
    windows=["7d", "30d"],
    aggregations=["mean", "sum"],
    include_current=False,
)
df_snapshot = snapshot.fit_transform(df_cyc)
df_snapshot.select([c for c in df_snapshot.columns if c.startswith("cnt__window")][:6]).head()

# %% [markdown]
# ## Exponentially weighted moving averages
#
# `EWMAggregator` reacts faster to recent changes than fixed windows. We compute an alpha=0.3 EWM
# grouped by the working-day flag.

# %%
ewm = EWMAggregator(
    time_column=time_column,
    groupby=["workingday"],
    value_columns=[target_column],
    alpha=0.3,
)
df_ewm = ewm.fit_transform(df_snapshot)
df_ewm.select([c for c in df_ewm.columns if c.endswith("__ewm")]).head()

# %% [markdown]
# ## Suggested automated time-series plan
#
# Provide `suggest_methods` with the time column and season grouping to inspect
# the default rolling-window recommendations.

# %%
suggested_plan = suggest_methods(
    raw_df,
    target_column=target_column,
    time_column=time_column,
    groupby=["season"],
)
suggested_plan

# %% [markdown]
# ## AutoFeaturizer with temporal context
#
# AutoFeaturizer can generate ranked, encoded, and aggregated features in one
# go, preserving the original signals for manual experimentation.

# %%
auto_featurizer = AutoFeaturizer(
    target_column=target_column,
    time_column=time_column,
    groupby=["season"],
    add_ranks=True,
    drop_original=False,
)
auto_df = auto_featurizer.fit_transform(raw_df)
auto_preview_cols = [target_column] + [c for c in auto_df.columns if c != target_column][:9]
auto_df.select(auto_preview_cols).head()

# %% [markdown]
# Between cyclical encodings, trailing windows, and EWM features, we now capture weekly and monthly
# trends alongside structural seasonality. Combine with weather-based interactions for even better
# forecasts, and use the AutoFeaturizer output as a baseline feature matrix.
