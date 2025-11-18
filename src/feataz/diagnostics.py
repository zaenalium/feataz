from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import polars as pl

from .base import _ensure_polars_df, _ensure_polars_series


def _prepare_cut_bins(bins: Sequence[float]) -> List[float]:
    """Return strictly increasing finite breakpoints compatible with polars.cut."""

    finite: List[float] = []
    for value in bins:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            continue
        finite.append(val)
    if not finite:
        return []
    return sorted(set(finite))


def information_value(
    df: pl.DataFrame,
    feature: str,
    target: str,
    bins: Optional[Sequence[float]] = None,
    eps: float = 1e-6,
) -> float:
    """Compute Information Value (IV) for a categorical or binned feature vs binary target.

    If `bins` is provided and the feature is numeric, the feature is cut into bins before IV.
    """
    df = _ensure_polars_df(df)

    if target not in df.columns or feature not in df.columns:
        raise ValueError("feature/target not found")
    s = df.get_column(feature)
    x = s
    if bins is not None and s.dtype.is_numeric():
        cut_bins = _prepare_cut_bins(bins)
        x = s.cast(pl.Float64).cut(cut_bins)
    temp = df.select([x.alias("feat"), pl.col(target).alias("target")])
    agg = (
        temp.group_by("feat")
        .agg([
            (pl.col("target") == 1).sum().alias("pos"),
            (pl.col("target") == 0).sum().alias("neg"),
        ])
        .with_columns([
            (pl.col("pos") / (pl.sum("pos") + eps)).alias("dist_pos"),
            (pl.col("neg") / (pl.sum("neg") + eps)).alias("dist_neg"),
            ((pl.col("pos") / (pl.sum("pos") + eps)) - (pl.col("neg") / (pl.sum("neg") + eps))).alias("diff"),
        ])
        .with_columns((pl.col("diff") * ((pl.col("dist_pos") + eps) / (pl.col("dist_neg") + eps)).log()).alias("iv_part"))
    )
    return float(agg.get_column("iv_part").sum())


def ks_statistic(df: pl.DataFrame, score: str, target: str) -> float:
    """Compute KS statistic for a score column vs binary target.

    KS = max_x |TPR(x) - FPR(x)| with TPR/FPR computed over the score-sorted records.
    """
    df = _ensure_polars_df(df)

    if score not in df.columns or target not in df.columns:
        raise ValueError("score/target not found")
    # Prepare sorted frame by score (ascending)
    temp = df.select([
        pl.col(score).cast(pl.Float64).alias("s"),
        pl.col(target).cast(pl.Int64).alias("y"),
    ]).sort("s")
    n_pos = float(temp.filter(pl.col("y") == 1).height)
    n_neg = float(temp.filter(pl.col("y") == 0).height)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # Cumulative positive/negative counts down the sorted list
    temp = temp.with_columns([
        pl.col("y").cum_sum().alias("cum_pos"),
        (pl.lit(1) - pl.col("y")).cum_sum().alias("cum_neg"),
    ])
    temp = temp.with_columns([
        (pl.col("cum_pos") / n_pos).alias("tpr"),
        (pl.col("cum_neg") / n_neg).alias("fpr"),
    ])
    ks = (temp.get_column("tpr") - temp.get_column("fpr")).abs().max()
    return float(ks)


def psi(
    expected: pl.Series,
    actual: pl.Series,
    bins: Optional[Sequence[float]] = None,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index between two distributions.

    If `bins` is None, determines bins by quantiles of expected.
    """
    expected = _ensure_polars_series(expected)
    actual = _ensure_polars_series(actual)

    if bins is None:
        qs = [i / n_bins for i in range(1, n_bins)]
        quantiles = pl.DataFrame({"q": [expected.quantile(q) for q in qs]}).get_column("q").to_list()
        edges = [float(q) for q in quantiles if q is not None]
    else:
        edges = list(bins)
    cut_edges = _prepare_cut_bins(edges)
    e_bin = expected.cast(pl.Float64).cut(cut_edges)
    a_bin = actual.cast(pl.Float64).cut(cut_edges)
    e_dist = pl.DataFrame({"b": e_bin}).group_by("b").len()
    a_dist = pl.DataFrame({"b": a_bin}).group_by("b").len()
    n_e = float(len(expected))
    n_a = float(len(actual))
    joined = e_dist.join(a_dist, on="b", how="full", suffix="_a").fill_null(0)
    joined = joined.with_columns([
        (pl.col("len") / (n_e + eps)).alias("p_e"),
        (pl.col("len_a") / (n_a + eps)).alias("p_a"),
    ])
    joined = joined.with_columns(
        ((pl.col("p_e") - pl.col("p_a")) * ((pl.col("p_e") + eps) / (pl.col("p_a") + eps)).log()).alias("psi_part")
    )
    return float(joined.get_column("psi_part").sum())
