feataz — Feature engineering on Polars
======================================

Lightweight feature engineering utilities on top of Polars DataFrames. Provides
categorical encoders, numerical discretizers, variance-stabilizing transforms,
and simple feature interactions with a consistent fit/transform API.

Install
-------

Install from PyPI:

    pip install feataz

Optional extras:

- Decision-tree based transformers: `pip install feataz[sklearn]`
- Accelerated similarity / numerical routines: `pip install feataz[accelerated]`

From a cloned checkout you can also install in editable mode:

    pip install -e .

Optional dependencies (only for tree-based transformers):

- DecisionTreeEncoder, DecisionTreeDiscretizer: `pip install scikit-learn`

Quick start
-----------

```python
import polars as pl
from feataz import (
    OneHotEncoder, OrdinalEncoder, CountFrequencyEncoder, MeanEncoder,
    WeightOfEvidenceEncoder, DecisionTreeEncoder, RareLabelEncoder, StringSimilarityEncoder,
    EqualFrequencyDiscretizer, EqualWidthDiscretizer, ArbitraryDiscretizer, DecisionTreeDiscretizer, GeometricWidthDiscretizer,
    LogTransformer, LogCPTransformer, ReciprocalTransformer, ArcsinTransformer, PowerTransformer, BoxCoxTransformer, YeoJohnsonTransformer,
    FeatureInteractions, TimeSnapshotAggregator, DynamicSnapshotAggregator, EWMAggregator, ToDateSnapshotAggregator,
    MathFeatures, RelativeFeatures, CyclicalFeatures, DecisionTreeFeatures, CrossFitTransformer,
    SimpleImputer, GroupImputer, KNNImputer, IterativeImputer, TimeSeriesImputer,
    ClipOutliers, IsolationForestOutlierHandler,
    RobustScaler, QuantileRankTransformer,
    HashEncoder, BinaryEncoder, LeaveOneOutEncoder,
    KMeansDiscretizer, MDLPDiscretizer, ChiMergeDiscretizer, IsotonicBinningDiscretizer, MonotonicOptimalBinningDiscretizer,
    information_value, ks_statistic, psi,
)

df = pl.DataFrame({
    "cat": ["a", "b", "a", "c", "b", "a"],
    "x": [1.0, 2.5, 0.5, 3.2, 2.7, 4.1],
    "y": [0, 1, 0, 1, 1, 0],
})

# Categorical encoders
df1 = OneHotEncoder(["cat"]).fit_transform(df)
df2 = OrdinalEncoder(["cat"], drop_original=False).fit_transform(df)
df3 = CountFrequencyEncoder(["cat"], normalize=True).fit_transform(df)
df4 = MeanEncoder(target="y", columns=["cat"], smoothing=5).fit_transform(df)
df5 = WeightOfEvidenceEncoder(target="y", columns=["cat"]).fit_transform(df)
df6 = RareLabelEncoder(["cat"], min_frequency=0.3, drop_original=False).fit_transform(df)
df7 = StringSimilarityEncoder(["cat"], top_k_anchors=2).fit_transform(df)

# Discretization
df8 = EqualFrequencyDiscretizer(["x"], n_bins=3).fit_transform(df)
df9 = EqualWidthDiscretizer(["x"], n_bins=4).fit_transform(df)
df10 = GeometricWidthDiscretizer(["x"], n_bins=3).fit_transform(df)
df11 = ArbitraryDiscretizer(bins=[0, 2, 4, float("inf")], columns=["x"]).fit_transform(df)

# Variance-stabilizing transforms
df12 = LogTransformer(["x"]).fit_transform(df)
df13 = LogCPTransformer(["x"], c=1.0).fit_transform(df)
df14 = ReciprocalTransformer(["x"]).fit_transform(df)
df15 = PowerTransformer(power=0.5, columns=["x"]).fit_transform(df)
df16 = BoxCoxTransformer(["x"]).fit_transform(df)
df17 = YeoJohnsonTransformer(["x"]).fit_transform(df)

# Interactions (group-based aggregations)
df18 = FeatureInteractions(groupby=["cat"], value_columns=["x"], aggregations=["sum", "mean"]).fit_transform(df)

# Time snapshots (trailing windows)
df_ts = pl.DataFrame({
    "cat": ["a","a","a","b","b"],
    "ts": pl.date_range(low=pl.datetime(2024,1,1), high=pl.datetime(2024,1,5), interval="1d", eager=True),
    "x": [1.0, 2.0, 3.0, 1.5, 2.5],
})
snap = TimeSnapshotAggregator(
    time_column="ts",
    groupby=["cat"],
    value_columns=["x"],
    windows=["3d", "1w"],
    aggregations=["sum", "mean"],
    include_current=False,
)
df19 = snap.fit_transform(df_ts)

# Math features (pairwise + unary)
df20 = MathFeatures(columns=["x"], unary_ops=["log","sqrt","abs"], powers=[2,3], binary_ops=("add","sub","mul","div")).fit_transform(df)

# Relative features (relative to a reference variable)
df21 = RelativeFeatures(target_columns=["x"], reference_columns=["y"], operations=["ratio","diff","pct_diff"]).fit_transform(
    pl.DataFrame({"x": [10,20,30], "y": [2,4,5]})
)

# Cyclical features (e.g., hour of day)
df22 = CyclicalFeatures(columns=["hour"], period=24).fit_transform(pl.DataFrame({"hour": [0,6,12,18]}))

# Decision tree features (predictions from a tree over multiple columns)
# Requires scikit-learn
# df_tree = DecisionTreeFeatures(target="y", columns=["x"], problem="auto", output="auto").fit_transform(df)

# Imputation (with per-column configs and indicators)
df_imp = pl.DataFrame({"cat": ["a", None, "b"], "x": [1.0, None, 3.0]})
# Simple: per-column strategy and indicator
df_imp1 = SimpleImputer(add_indicator=True, numeric_strategy_map={"x": "mean"}).fit_transform(df_imp)
# Group-based (by cat)
df_imp2 = GroupImputer(groupby=["cat"], columns=["x"], numeric_strategy="mean").fit_transform(df_imp)
# Time-series forward/backward fill per group
df_tsi = TimeSeriesImputer(time_column="ts", columns=["x"], groupby=["cat"], method="both")

# Advanced (sklearn): KNN and Iterative
# df_imp_knn = KNNImputer(columns=["x"], n_neighbors=3).fit_transform(df_imp)
# df_imp_mice = IterativeImputer(columns=["x"], max_iter=5).fit_transform(df_imp)

# Outlier handling
df_out = pl.DataFrame({"x": [1, 2, 3, 100, -50], "y": [0, 0, 1, 1, 0]})
# Simple clipping with quantiles
df_out1 = ClipOutliers(columns=["x"], method="quantile", q_low=0.05, q_high=0.95, action="clip").fit_transform(df_out)
# IQR flags
df_out2 = ClipOutliers(columns=["x"], method="iqr", iqr_factor=1.5, action="flag").fit_transform(df_out)
# Advanced (sklearn): Isolation Forest
# df_out3 = IsolationForestOutlierHandler(columns=["x"], contamination=0.1, action="flag", add_score=True).fit_transform(df_out)

# Advanced encoders
df_hash = HashEncoder(columns=["cat"], n_components=16).fit_transform(df)
df_bin = BinaryEncoder(columns=["cat"]).fit_transform(df)
df_loo = LeaveOneOutEncoder(target="y", columns=["cat"], smoothing=5).fit_transform(df)

# Cross-fitted target encoders (leakage-safe)
# oof = CrossFitTransformer(lambda: MeanEncoder(target="y", columns=["cat"], smoothing=5), n_splits=5).fit(df)
# df_oof = oof._oof_train_

# KMeans (sklearn), MDLP, ChiMerge, Isotonic, and Monotonic Optimal (supervised)
# df_km = KMeansDiscretizer(columns=["x"], n_bins=3).fit_transform(df)
# df_mdlp = MDLPDiscretizer(target="y", columns=["x"], min_samples_bin=10).fit_transform(df)
# df_chm = ChiMergeDiscretizer(target="y", columns=["x"], max_bins=6, alpha=0.05).fit_transform(df)
# df_iso = IsotonicBinningDiscretizer(target="y", columns=["x"], monotonic="auto", max_bins=6).fit_transform(df)
# df_mob = MonotonicOptimalBinningDiscretizer(target="y", columns=["x"], initial_bins=50, min_bin_size=0.05, max_bins=6).fit_transform(df)

# Dynamic snapshots (calendar windows), To-Date, and EWM mean
dyn = DynamicSnapshotAggregator(time_column="ts", every="1w", groupby=["cat"], value_columns=["x"], aggregations=["sum"]).fit(df_ts)
td = ToDateSnapshotAggregator(time_column="ts", period="1mo", groupby=["cat"], value_columns=["x"], aggregations=["sum","mean"], include_current=False).fit(df_ts)
ewm = EWMAggregator(time_column="ts", alpha=0.5, groupby=["cat"], value_columns=["x"]).fit(df_ts)
# df_dyn = dyn.transform(df_ts)
# df_todate = td.transform(df_ts)
# df_ewm = ewm.transform(df_ts)

# Robust scaling and rank
df_rs = RobustScaler(columns=["x"]).fit_transform(df)
df_rk = QuantileRankTransformer(columns=["x"], groupby=["cat"]).fit_transform(df)

# Diagnostics
# iv = information_value(df18, feature="cat__qbin", target="y")
# ks = ks_statistic(df, score="score", target="y")
# s = psi(expected=df_baseline.get_column("score"), actual=df_current.get_column("score"))
```

Notes
-----

- Tree-based encoders/discretizers require scikit-learn and operate per-column using a univariate tree.
- Box-Cox and Yeo-Johnson lambdas are estimated via a lightweight grid search, no SciPy required.
- String similarity uses Python’s `difflib.SequenceMatcher`; provide anchors or it picks top-k frequent values per column.
- Transformations default to returning new columns with suffixes and typically drop originals; configure via constructor args.
