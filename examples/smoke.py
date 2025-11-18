import polars as pl
from feataz import (
    OneHotEncoder,
    OrdinalEncoder,
    CountFrequencyEncoder,
    MeanEncoder,
    WeightOfEvidenceEncoder,
    RareLabelEncoder,
    StringSimilarityEncoder,
    EqualFrequencyDiscretizer,
    EqualWidthDiscretizer,
    GeometricWidthDiscretizer,
    ArbitraryDiscretizer,
    LogTransformer,
    LogCPTransformer,
    ReciprocalTransformer,
    PowerTransformer,
    BoxCoxTransformer,
    YeoJohnsonTransformer,
    FeatureInteractions,
    TimeSnapshotAggregator,
    MathFeatures,
    RelativeFeatures,
    CyclicalFeatures,
    DecisionTreeFeatures,
    SimpleImputer,
    GroupImputer,
    KNNImputer,
    IterativeImputer,
    TimeSeriesImputer,
    ClipOutliers,
    IsolationForestOutlierHandler,
    RobustScaler,
    QuantileRankTransformer,
    HashEncoder,
    BinaryEncoder,
    LeaveOneOutEncoder,
    KMeansDiscretizer,
    MDLPDiscretizer,
    ChiMergeDiscretizer,
    IsotonicBinningDiscretizer,
    MonotonicOptimalBinningDiscretizer,
    CrossFitTransformer,
    DynamicSnapshotAggregator,
    EWMAggregator,
    ToDateSnapshotAggregator,
)


def main():
    df = pl.DataFrame({
        "cat": ["a", "b", "a", "c", "b", "a"],
        "x": [1.0, 2.5, 0.5, 3.2, 2.7, 4.1],
        "y": [0, 1, 0, 1, 1, 0],
    })
    print("Input:\n", df)

    df1 = OneHotEncoder(["cat"]).fit_transform(df)
    print("\nOneHot:\n", df1)

    df2 = OrdinalEncoder(["cat"], drop_original=False).fit_transform(df)
    print("\nOrdinal:\n", df2)

    df3 = CountFrequencyEncoder(["cat"], normalize=True).fit_transform(df)
    print("\nCountFreq:\n", df3)

    df4 = MeanEncoder(target="y", columns=["cat"], smoothing=2).fit_transform(df)
    print("\nMeanEnc:\n", df4)

    df5 = WeightOfEvidenceEncoder(target="y", columns=["cat"]).fit_transform(df)
    print("\nWOE:\n", df5)

    df6 = RareLabelEncoder(["cat"], min_frequency=0.34, drop_original=False).fit_transform(df)
    print("\nRareLabel:\n", df6)

    df7 = StringSimilarityEncoder(["cat"], top_k_anchors=2).fit_transform(df)
    print("\nStringSimilarity:\n", df7)

    df8 = EqualFrequencyDiscretizer(["x"], n_bins=3).fit_transform(df)
    print("\nEqualFreq:\n", df8)

    df9 = EqualWidthDiscretizer(["x"], n_bins=3).fit_transform(df)
    print("\nEqualWidth:\n", df9)

    df10 = GeometricWidthDiscretizer(["x"], n_bins=3).fit_transform(df)
    print("\nGeomWidth:\n", df10)

    df11 = ArbitraryDiscretizer(bins=[0, 2, 4, float("inf")], columns=["x"]).fit_transform(df)
    print("\nArbitrary:\n", df11)

    df12 = LogTransformer(["x"]).fit_transform(df)
    print("\nLog:\n", df12)

    df13 = LogCPTransformer(["x"], c=1.0).fit_transform(df)
    print("\nLogCP:\n", df13)

    df14 = ReciprocalTransformer(["x"]).fit_transform(df)
    print("\nReciprocal:\n", df14)

    df15 = PowerTransformer(power=0.5, columns=["x"]).fit_transform(df)
    print("\nPower 0.5:\n", df15)

    df16 = BoxCoxTransformer(["x"]).fit_transform(df)
    print("\nBoxCox:\n", df16)

    df17 = YeoJohnsonTransformer(["x"]).fit_transform(df)
    print("\nYeoJohnson:\n", df17)

    df18 = FeatureInteractions(groupby=["cat"], value_columns=["x"], aggregations=["sum", "mean"]).fit_transform(df)
    print("\nGroup Interactions (sum/mean x by cat):\n", df18)

    # Snapshot features on a time series
    df_ts = pl.DataFrame({
        "cat": ["a","a","a","b","b"],
        "ts": pl.date_range(low=pl.datetime(2024,1,1), high=pl.datetime(2024,1,5), interval="1d", eager=True),
        "x": [1.0, 2.0, 3.0, 1.5, 2.5],
    })
    snap = TimeSnapshotAggregator(
        time_column="ts",
        groupby=["cat"],
        value_columns=["x"],
        windows=["3d"],
        aggregations=["sum", "mean"],
        include_current=False,
    )
    df19 = snap.fit_transform(df_ts)
    print("\nSnapshot features (last 3d sum/mean of x by cat):\n", df19)

    # Math features
    df20 = MathFeatures(columns=["x"], unary_ops=["log","sqrt","abs"], powers=[2], binary_ops=("add",)).fit_transform(df)
    print("\nMath features (x log/sqrt/abs, x^2, and x add pairs if present):\n", df20)

    # Relative features
    df_rel = pl.DataFrame({"a": [10, 20, 30], "b": [2, 4, 5]})
    df21 = RelativeFeatures(target_columns=["a"], reference_columns=["b"], operations=["ratio","diff","pct_diff"]).fit_transform(df_rel)
    print("\nRelative features a vs b (ratio/diff/pctdiff):\n", df21)

    # Cyclical features
    df_cyc = pl.DataFrame({"hour": [0, 6, 12, 18]})
    df22 = CyclicalFeatures(columns=["hour"], period=24).fit_transform(df_cyc)
    print("\nCyclical features for hour (sin/cos):\n", df22)

    # Decision tree features (requires scikit-learn)
    try:
        df_tree = DecisionTreeFeatures(target="y", columns=["x"], problem="auto", output="auto").fit_transform(df)
        print("\nDecision tree feature over x predicting y:\n", df_tree)
    except Exception as e:
        print("\nSkipping DecisionTreeFeatures due to missing sklearn:", e)

    # Imputation examples
    df_imp = pl.DataFrame({"cat": ["a", None, "b"], "x": [1.0, None, 3.0]})
    df_imp1 = SimpleImputer().fit_transform(df_imp)
    print("\nSimpleImputer (median/mode):\n", df_imp1)

    df_imp2 = GroupImputer(groupby=["cat"], columns=["x"], numeric_strategy="mean").fit_transform(df_imp)
    print("\nGroupImputer (mean of x by cat):\n", df_imp2)

    # Time-series imputation
    df_ts2 = pl.DataFrame({
        "cat": ["a","a","a","b","b"],
        "ts": pl.date_range(low=pl.datetime(2024,1,1), high=pl.datetime(2024,1,5), interval="1d", eager=True),
        "x": [1.0, None, 3.0, None, 2.0],
    })
    df_imp3 = TimeSeriesImputer(time_column="ts", columns=["x"], groupby=["cat"], method="both").fit_transform(df_ts2)
    print("\nTimeSeriesImputer (both) for x by cat:\n", df_imp3)

    # Advanced imputers (sklearn) guarded
    try:
        df_imp_knn = KNNImputer(columns=["x"], n_neighbors=2).fit_transform(df_imp)
        print("\nKNNImputer on x:\n", df_imp_knn)
    except Exception as e:
        print("\nSkipping KNNImputer due to missing sklearn:", e)
    try:
        df_imp_mice = IterativeImputer(columns=["x"], max_iter=3).fit_transform(df_imp)
        print("\nIterativeImputer (MICE) on x:\n", df_imp_mice)
    except Exception as e:
        print("\nSkipping IterativeImputer due to missing sklearn:", e)

    # Outlier handling
    df_out = pl.DataFrame({"x": [1, 2, 3, 100, -50], "y": [0, 0, 1, 1, 0]})
    df_out1 = ClipOutliers(columns=["x"], method="quantile", q_low=0.05, q_high=0.95, action="clip").fit_transform(df_out)
    print("\nClipOutliers quantile clip on x:\n", df_out1)

    df_out2 = ClipOutliers(columns=["x"], method="iqr", iqr_factor=1.5, action="flag").fit_transform(df_out)
    print("\nClipOutliers IQR flag on x:\n", df_out2)

    try:
        df_out3 = IsolationForestOutlierHandler(columns=["x"], contamination=0.4, action="flag", add_score=True).fit_transform(df_out)
        print("\nIsolationForest outlier flags/scores:\n", df_out3)
    except Exception as e:
        print("\nSkipping IsolationForestOutlierHandler due to missing sklearn:", e)

    # Advanced encoders
    df_hash = HashEncoder(columns=["cat"], n_components=8).fit_transform(df)
    print("\nHashEncoder(cat -> 8 buckets):\n", df_hash)
    df_bin = BinaryEncoder(columns=["cat"], drop_original=False).fit_transform(df)
    print("\nBinaryEncoder(cat -> bits):\n", df_bin)
    df_loo = LeaveOneOutEncoder(target="y", columns=["cat"], smoothing=2).fit_transform(df)
    print("\nLeaveOneOutEncoder(cat vs y):\n", df_loo)

    # Cross-fitted encoder (OOF) using MeanEncoder
    try:
        oof = CrossFitTransformer(lambda: MeanEncoder(target="y", columns=["cat"], smoothing=2), n_splits=3)
        oof.fit(df)
        print("\nCrossFitTransformer OOF columns:", getattr(oof, "oof_columns_", []))
    except Exception as e:
        print("\nCrossFitTransformer example error:", e)

    # KMeans discretizer
    try:
        df_km = KMeansDiscretizer(columns=["x"], n_bins=3).fit_transform(df)
        print("\nKMeansDiscretizer on x:\n", df_km)
    except Exception as e:
        print("\nSkipping KMeansDiscretizer due to missing sklearn:", e)

    # Supervised discretizers on x vs y (binary target)
    try:
        df_mdlp = MDLPDiscretizer(target="y", columns=["x"], min_samples_bin=2).fit_transform(df)
        print("\nMDLPDiscretizer (x vs y):\n", df_mdlp)
    except Exception as e:
        print("\nMDLPDiscretizer error:", e)

    df_chm = ChiMergeDiscretizer(target="y", columns=["x"], max_bins=4, alpha=0.05, initial_bins=5, min_samples_bin=1).fit_transform(df)
    print("\nChiMergeDiscretizer (x vs y):\n", df_chm)

    try:
        df_iso = IsotonicBinningDiscretizer(target="y", columns=["x"], monotonic="auto", max_bins=4, min_samples_bin=1).fit_transform(df)
        print("\nIsotonicBinningDiscretizer (x vs y):\n", df_iso)
    except Exception as e:
        print("\nSkipping IsotonicBinningDiscretizer due to missing sklearn:", e)

    df_mob = MonotonicOptimalBinningDiscretizer(target="y", columns=["x"], initial_bins=10, min_bin_size=1, max_bins=4).fit_transform(df)
    print("\nMonotonicOptimalBinningDiscretizer (x vs y):\n", df_mob)

    # Dynamic snapshots and EWM
    dyn = DynamicSnapshotAggregator(time_column="ts", every="1w", groupby=["cat"], value_columns=["x"], aggregations=["sum"]).fit(df_ts)
    df_dyn = dyn.transform(df_ts)
    print("\nDynamic weekly sum of x by cat:\n", df_dyn)
    td = ToDateSnapshotAggregator(time_column="ts", period="1mo", groupby=["cat"], value_columns=["x"], aggregations=["sum","mean"], include_current=False).fit(df_ts)
    df_td = td.transform(df_ts)
    print("\nToDate (MTD) sum/mean of x by cat (excluding current):\n", df_td)
    ewm = EWMAggregator(time_column="ts", alpha=0.5, groupby=["cat"], value_columns=["x"]).fit(df_ts)
    df_ewm = ewm.transform(df_ts)
    print("\nEWM mean of x by cat (alpha=0.5):\n", df_ewm)

    # Scaling & rank
    df_rs = RobustScaler(columns=["x"]).fit_transform(df)
    print("\nRobustScaler on x:\n", df_rs)
    df_rk = QuantileRankTransformer(columns=["x"], groupby=["cat"]).fit_transform(df)
    print("\nQuantileRankTransformer of x within cat:\n", df_rk)


if __name__ == "__main__":
    main()
