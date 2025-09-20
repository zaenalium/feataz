import pytest

pl = pytest.importorskip("polars", reason="polars is required for diagnostics tests")

if not hasattr(pl.datatypes, "is_numeric"):
    def _is_numeric(dtype: pl.DataType) -> bool:  # type: ignore[name-defined]
        try:
            return bool(dtype.is_numeric())  # type: ignore[attr-defined]
        except AttributeError:
            return False

    pl.datatypes.is_numeric = _is_numeric  # type: ignore[attr-defined]

if not hasattr(pl.datatypes, "is_string_dtype"):
    def _is_string(dtype: pl.DataType) -> bool:  # type: ignore[name-defined]
        try:
            return bool(dtype.is_(pl.Utf8) or dtype.is_(pl.String))  # type: ignore[attr-defined]
        except AttributeError:
            return False

    pl.datatypes.is_string_dtype = _is_string  # type: ignore[attr-defined]

from feataz.diagnostics import information_value, ks_statistic, psi

EPS = 1e-6


def _quantile_edges(series: pl.Series, segments: int) -> list[float]:
    probs = [i / segments for i in range(1, segments)]
    numeric = series.cast(pl.Float64)
    quantiles = [numeric.quantile(q) for q in probs]
    edges = [-float("inf")]
    for value in quantiles:
        if value is None:
            continue
        val = float(value)
        if not edges or val > edges[-1]:
            edges.append(val)
    edges.append(float("inf"))
    return edges


def test_information_value_wine(wine_binary: pl.DataFrame) -> None:
    feature = "alcohol"
    feature_series = wine_binary.get_column(feature).cast(pl.Float64)
    edges = _quantile_edges(feature_series, 4)

    manual = (
        wine_binary.with_columns(pl.col(feature).cut(edges).alias("bin"))
        .group_by("bin")
        .agg([
            (pl.col("target_bin") == 1).sum().alias("pos"),
            (pl.col("target_bin") == 0).sum().alias("neg"),
        ])
        .with_columns([
            (pl.col("pos") / (pl.sum("pos") + EPS)).alias("dist_pos"),
            (pl.col("neg") / (pl.sum("neg") + EPS)).alias("dist_neg"),
        ])
        .with_columns(
            (
                (pl.col("dist_pos") - pl.col("dist_neg"))
                * ((pl.col("dist_pos") + EPS) / (pl.col("dist_neg") + EPS)).log()
            ).alias("iv_part")
        )
    )
    expected_iv = float(manual.get_column("iv_part").sum())
    result_iv = information_value(wine_binary, feature, "target_bin", bins=edges, eps=EPS)

    assert result_iv == pytest.approx(expected_iv, rel=1e-9, abs=1e-9)
    assert result_iv >= 0.0


def test_ks_statistic_wine(wine_binary: pl.DataFrame) -> None:
    color_series = wine_binary.get_column("color_intensity").cast(pl.Float64)
    max_color = float(color_series.max())
    scored = wine_binary.with_columns(
        (pl.col("color_intensity") / max_color).alias("score")
    )

    result_ks = ks_statistic(scored, "score", "target_bin")

    manual = scored.select([
        pl.col("score").cast(pl.Float64).alias("s"),
        pl.col("target_bin").cast(pl.Int64).alias("y"),
    ]).sort("s")
    n_pos = float(manual.filter(pl.col("y") == 1).height)
    n_neg = float(manual.filter(pl.col("y") == 0).height)
    assert n_pos > 0 and n_neg > 0

    manual = manual.with_columns([
        pl.col("y").cum_sum().alias("cum_pos"),
        (pl.lit(1) - pl.col("y")).cum_sum().alias("cum_neg"),
    ]).with_columns([
        (pl.col("cum_pos") / n_pos).alias("tpr"),
        (pl.col("cum_neg") / n_neg).alias("fpr"),
    ])
    expected_ks = float((manual.get_column("tpr") - manual.get_column("fpr")).abs().max())

    assert result_ks == pytest.approx(expected_ks, rel=1e-9, abs=1e-9)
    assert 0.0 <= result_ks <= 1.0


def test_population_stability_index_wine(wine_df: pl.DataFrame) -> None:
    expected_series = wine_df.filter(pl.col("target") == 0).get_column("alcohol")
    actual_series = wine_df.filter(pl.col("target") == 1).get_column("alcohol")

    edges = _quantile_edges(expected_series.cast(pl.Float64), 8)
    result_psi = psi(expected_series, actual_series, bins=edges, eps=EPS)

    e_bin = expected_series.cast(pl.Float64).cut(edges)
    a_bin = actual_series.cast(pl.Float64).cut(edges)
    e_dist = pl.DataFrame({"b": e_bin}).group_by("b").len()
    a_dist = pl.DataFrame({"b": a_bin}).group_by("b").len()
    n_e = float(len(expected_series))
    n_a = float(len(actual_series))

    manual = e_dist.join(a_dist, on="b", how="outer", suffix="_a").fill_null(0).with_columns([
        (pl.col("len") / (n_e + EPS)).alias("p_e"),
        (pl.col("len_a") / (n_a + EPS)).alias("p_a"),
        ((pl.col("p_e") - pl.col("p_a")) * ((pl.col("p_e") + EPS) / (pl.col("p_a") + EPS)).log()).alias("psi_part"),
    ])
    expected_psi = float(manual.get_column("psi_part").sum())

    assert result_psi == pytest.approx(expected_psi, rel=1e-9, abs=1e-9)
    assert result_psi >= 0.0
