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


if __name__ == "__main__":
    main()
