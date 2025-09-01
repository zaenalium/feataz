from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import math
import polars as pl

from .base import Transformer, _ensure_polars_df


def _infer_numeric_columns(df: pl.DataFrame, columns: Optional[Sequence[str]]) -> List[str]:
    if columns is not None:
        return list(columns)
    nums: List[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if pl.datatypes.is_numeric(dtype):
            nums.append(name)
    return nums


def _make_bins_safe(edges: List[float]) -> List[float]:
    # ensure sorted and strictly increasing (dedupe)
    edges_sorted = sorted(set(float(e) for e in edges))
    # if not enough edges, duplicate min/max
    if len(edges_sorted) < 2:
        mn = edges_sorted[0] if edges_sorted else -float("inf")
        mx = edges_sorted[0] if edges_sorted else float("inf")
        return [mn, mx]
    return edges_sorted


class EqualFrequencyDiscretizer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        n_bins: int = 5,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__qbin",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.n_bins = int(n_bins)
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}

    def fit(self, df: pl.DataFrame) -> "EqualFrequencyDiscretizer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.bins_.clear()
        qs = [i / self.n_bins for i in range(1, self.n_bins)]
        for col in cols:
            quantiles = df.select([pl.col(col).quantile(q) for q in qs]).row(0)
            edges = [-float("inf")] + [float(q) for q in quantiles] + [float("inf")]
            self.bins_[col] = _make_bins_safe(edges)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            edges = self.bins_[col]
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(
                pl.col(col).cast(pl.Float64).cut(bins=edges, labels=labels).alias(new_col)
            )
            if self.drop_original:
                out = out.drop(col)
        return out


class EqualWidthDiscretizer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        n_bins: int = 5,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__wbin",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.n_bins = int(n_bins)
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}

    def fit(self, df: pl.DataFrame) -> "EqualWidthDiscretizer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.bins_.clear()
        for col in cols:
            s = df.get_column(col).cast(pl.Float64)
            mn = float(s.min())
            mx = float(s.max())
            if math.isfinite(mn) and math.isfinite(mx) and mx > mn:
                step = (mx - mn) / self.n_bins
                edges = [mn + i * step for i in range(self.n_bins + 1)]
            else:
                edges = [mn, mx]
            edges[0] = -float("inf")
            edges[-1] = float("inf")
            self.bins_[col] = _make_bins_safe(edges)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            edges = self.bins_[col]
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class ArbitraryDiscretizer(Transformer):
    def __init__(
        self,
        bins: Sequence[float] | Dict[str, Sequence[float]],
        columns: Optional[Sequence[str]] = None,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__abin",
    ) -> None:
        self.bins_input = bins
        self.columns = None if columns is None else list(columns)
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}

    def fit(self, df: pl.DataFrame) -> "ArbitraryDiscretizer":
        df = _ensure_polars_df(df)
        if isinstance(self.bins_input, dict):
            mapping = {k: _make_bins_safe(list(v)) for k, v in self.bins_input.items()}
            cols = list(mapping.keys()) if self.columns is None else list(self.columns)
            self.bins_ = {c: mapping[c] for c in cols}
        else:
            cols = _infer_numeric_columns(df, self.columns)
            edges = _make_bins_safe(list(self.bins_input))
            self.bins_ = {c: edges for c in cols}
        self.feature_names_in_ = list(self.bins_.keys())
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col, edges in self.bins_.items():
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class DecisionTreeDiscretizer(Transformer):
    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        problem: str = "auto",
        max_leaf_nodes: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__dtbin",
        random_state: Optional[int] = 42,
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.problem = problem
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.random_state = random_state
        self.bins_: Dict[str, List[float]] = {}

    def fit(self, df: pl.DataFrame) -> "DecisionTreeDiscretizer":
        try:
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "DecisionTreeDiscretizer requires scikit-learn. Install with `pip install scikit-learn`."
            ) from e

        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        y = df.get_column(self.target).to_numpy()
        # Determine problem if auto
        problem = self.problem
        if problem == "auto":
            n_unique = len(pl.Series(y).drop_nulls().unique())
            if n_unique <= 20 and pl.Series(y).dtype in (pl.Int64, pl.Int32, pl.UInt32, pl.UInt64, pl.Boolean):
                problem = "classification"
            else:
                problem = "regression"
        self.bins_.clear()
        for col in cols:
            X = df.get_column(col).to_numpy().reshape(-1, 1)
            if problem == "classification":
                tree = DecisionTreeClassifier(
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
            else:
                tree = DecisionTreeRegressor(
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
            tree.fit(X, y)
            thresholds = [t for t in getattr(tree.tree_, "threshold").tolist() if t > -2]
            edges = [-float("inf")] + sorted(set(float(t) for t in thresholds)) + [float("inf")]
            self.bins_[col] = _make_bins_safe(edges)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col, edges in self.bins_.items():
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class GeometricWidthDiscretizer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        n_bins: int = 5,
        drop_original: bool = True,
        labels_as_int: bool = True,
        epsilon: float = 1e-9,
        suffix: str = "__gbin",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.n_bins = int(n_bins)
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.epsilon = float(epsilon)
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}
        self.offsets_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame) -> "GeometricWidthDiscretizer":
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.bins_.clear()
        self.offsets_.clear()
        for col in cols:
            s = df.get_column(col).cast(pl.Float64)
            mn = float(s.min())
            mx = float(s.max())
            offset = 0.0
            if mn <= 0:
                offset = -mn + self.epsilon
            mn_p = mn + offset
            mx_p = mx + offset
            mn_p = max(mn_p, self.epsilon)
            mx_p = max(mx_p, mn_p + self.epsilon)
            r = (mx_p / mn_p) ** (1.0 / self.n_bins)
            edges_pos = [mn_p * (r**i) for i in range(self.n_bins + 1)]
            edges = [e - offset for e in edges_pos]
            edges[0] = -float("inf")
            edges[-1] = float("inf")
            self.bins_[col] = _make_bins_safe(edges)
            self.offsets_[col] = offset
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col, edges in self.bins_.items():
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class KMeansDiscretizer(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        n_bins: int = 5,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__kmbin",
        random_state: Optional[int] = 42,
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.n_bins = int(n_bins)
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.random_state = random_state
        self.centers_: Dict[str, List[float]] = {}

    def fit(self, df: pl.DataFrame) -> "KMeansDiscretizer":
        try:
            from sklearn.cluster import KMeans
        except Exception as e:
            raise ImportError("KMeansDiscretizer requires scikit-learn") from e
        df = _ensure_polars_df(df)
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.centers_.clear()
        for col in cols:
            X = df.get_column(col).cast(pl.Float64).to_numpy().reshape(-1, 1)
            km = KMeans(n_clusters=self.n_bins, random_state=self.random_state, n_init=10)
            km.fit(X)
            centers = sorted(km.cluster_centers_.flatten().tolist())
            self.centers_[col] = centers
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            import numpy as np
        except Exception:
            raise RuntimeError("NumPy required for KMeansDiscretizer transform")
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            centers = self.centers_[col]
            # create edges halfway between centers
            edges = [-float("inf")] + [float((centers[i] + centers[i + 1]) / 2.0) for i in range(len(centers) - 1)] + [float("inf")]
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class MDLPDiscretizer(Transformer):
    """Supervised MDLP (Fayyad-Irani) discretizer for a binary target.

    Recursively finds thresholds that maximize information gain subject to the MDL stopping criterion.
    """

    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        min_samples_bin: int = 20,
        max_bins: Optional[int] = None,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__mdlp",
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.min_samples_bin = int(min_samples_bin)
        self.max_bins = max_bins
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}

    @staticmethod
    def _entropy(pos: int, neg: int) -> float:
        import math
        n = pos + neg
        if n == 0:
            return 0.0
        p1 = pos / n
        p0 = neg / n
        def h(p):
            return 0.0 if p <= 0.0 or p >= 1.0 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        return h(p1)

    def _best_split(self, x: list[float], y: list[int]) -> tuple[float | None, float, tuple[int, int, int, int]]:
        # x sorted with corresponding y
        n = len(x)
        # candidate thresholds at midpoints where class changes
        best_gain = -1.0
        best_thr: float | None = None
        best_counts = (0, 0, 0, 0)
        # total counts
        pos_total = sum(y)
        neg_total = n - pos_total
        H = self._entropy(pos_total, neg_total)
        for i in range(1, n):
            if y[i] == y[i - 1]:
                continue
            thr = (x[i] + x[i - 1]) / 2.0
            pos_left = sum(y[:i])
            neg_left = i - pos_left
            pos_right = pos_total - pos_left
            neg_right = neg_total - neg_left
            n_left = i
            n_right = n - i
            if n_left < self.min_samples_bin or n_right < self.min_samples_bin:
                continue
            H_left = self._entropy(pos_left, neg_left)
            H_right = self._entropy(pos_right, neg_right)
            gain = H - (n_left / n) * H_left - (n_right / n) * H_right
            if gain > best_gain:
                best_gain = gain
                best_thr = thr
                best_counts = (pos_left, neg_left, pos_right, neg_right)
        return best_thr, best_gain, best_counts

    @staticmethod
    def _mdl_stop(gain: float, counts_parent: tuple[int, int], counts_left: tuple[int, int], counts_right: tuple[int, int], n: int) -> bool:
        # Fayyad & Irani MDL stopping criterion
        import math
        k = sum(1 for c in counts_parent if c > 0)  # number of classes in parent (binary)
        k1 = sum(1 for c in counts_left if c > 0)
        k2 = sum(1 for c in counts_right if c > 0)
        H = lambda a, b: 0.0 if (a + b) == 0 else (-(a/(a+b))*math.log2(a/(a+b)) - (b/(a+b))*math.log2(b/(a+b))) if 0 < a < (a+b) else 0.0
        delta = math.log2(3**k - 2) - (k * H(*counts_parent) - k1 * H(*counts_left) - k2 * H(*counts_right))
        threshold = (math.log2(n - 1) + delta) / n
        return gain <= threshold

    def _mdlp_splits(self, x: list[float], y: list[int], depth: int = 0) -> List[float]:
        n = len(x)
        if n < 2 * self.min_samples_bin:
            return []
        thr, gain, (pl, nl, pr, nr) = self._best_split(x, y)
        if thr is None:
            return []
        # MDL stop check
        if self._mdl_stop(gain, (pl + pr, nl + nr), (pl, nl), (pr, nr), n):
            return []
        # recurse
        left_mask = [xi <= thr for xi in x]
        x_left = [xi for xi, m in zip(x, left_mask) if m]
        y_left = [yi for yi, m in zip(y, left_mask) if m]
        x_right = [xi for xi, m in zip(x, left_mask) if not m]
        y_right = [yi for yi, m in zip(y, left_mask) if not m]
        splits = self._mdlp_splits(x_left, y_left, depth + 1) + [thr] + self._mdlp_splits(x_right, y_right, depth + 1)
        # optional cap on bins
        if self.max_bins is not None and (len(splits) + 1) > self.max_bins:
            # trim by keeping largest-gain central thresholds (approx: sort thresholds and keep evenly)
            splits = sorted(splits)[: self.max_bins - 1]
        return sorted(splits)

    def fit(self, df: pl.DataFrame) -> "MDLPDiscretizer":
        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.bins_.clear()
        y = df.get_column(self.target).cast(pl.Int64).to_list()
        if len(set([v for v in y if v is not None])) > 2:
            raise ValueError("MDLPDiscretizer currently supports binary target only")
        for col in cols:
            s = df.get_column(col).cast(pl.Float64)
            pairs = sorted([(float(a), int(b)) for a, b in zip(s.to_list(), y) if a is not None and b is not None], key=lambda t: t[0])
            if not pairs:
                self.bins_[col] = [-float("inf"), float("inf")]
                continue
            x_sorted = [p[0] for p in pairs]
            y_sorted = [p[1] for p in pairs]
            thresholds = self._mdlp_splits(x_sorted, y_sorted)
            edges = [-float("inf")] + thresholds + [float("inf")]
            self.bins_[col] = _make_bins_safe(edges)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col, edges in self.bins_.items():
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class ChiMergeDiscretizer(Transformer):
    """Supervised ChiMerge discretization for binary targets.

    Merges adjacent bins with the smallest chi-square until the minimum
    chi-square exceeds the critical value or `max_bins` is reached.
    """

    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        max_bins: int = 6,
        alpha: float = 0.05,
        initial_bins: int = 50,
        min_samples_bin: int = 10,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__chm",
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.max_bins = int(max_bins)
        self.alpha = float(alpha)
        self.initial_bins = int(initial_bins)
        self.min_samples_bin = int(min_samples_bin)
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}

    @staticmethod
    def _chi2_threshold(alpha: float) -> float:
        # df=1 critical values
        table = [(0.10, 2.706), (0.05, 3.841), (0.02, 5.412), (0.01, 6.635)]
        # choose closest by alpha
        best = min(table, key=lambda t: abs(t[0] - alpha))
        return best[1]

    @staticmethod
    def _chi2_pair(pos_l: int, neg_l: int, pos_r: int, neg_r: int, eps: float = 1e-9) -> float:
        n_l = pos_l + neg_l
        n_r = pos_r + neg_r
        n = n_l + n_r
        if n_l == 0 or n_r == 0:
            return 0.0
        tot_pos = pos_l + pos_r
        tot_neg = neg_l + neg_r
        E_lp = n_l * tot_pos / (n + eps)
        E_ln = n_l * tot_neg / (n + eps)
        E_rp = n_r * tot_pos / (n + eps)
        E_rn = n_r * tot_neg / (n + eps)
        chi = 0.0
        for O, E in (
            (pos_l, E_lp),
            (neg_l, E_ln),
            (pos_r, E_rp),
            (neg_r, E_rn),
        ):
            if E > 0:
                chi += (O - E) ** 2 / E
        return float(chi)

    def fit(self, df: pl.DataFrame) -> "ChiMergeDiscretizer":
        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        y = df.get_column(self.target).cast(pl.Int64)
        if df.select(pl.col(self.target).drop_nulls().n_unique()).item() > 2:
            raise ValueError("ChiMergeDiscretizer supports binary target only")
        crit = self._chi2_threshold(self.alpha)
        for col in cols:
            s = df.get_column(col).cast(pl.Float64)
            # initial edges by quantiles
            n_init = max(2, min(self.initial_bins, s.drop_nulls().n_unique()))
            qs = [i / n_init for i in range(1, n_init)]
            quantiles = df.select([pl.col(col).quantile(q) for q in qs]).row(0)
            edges = [-float("inf")] + [float(q) for q in quantiles] + [float("inf")]
            # build initial pos/neg per bin
            tmp = df.select([
                pl.col(col).cast(pl.Float64).cut(edges, labels=list(range(len(edges) - 1))).alias("__bin__"),
                y.alias("__y__"),
            ])
            agg = tmp.group_by("__bin__").agg([
                pl.len().alias("n"),
                pl.col("__y__").sum().alias("pos"),
            ]).sort("__bin__")
            n_list = agg.get_column("n").to_list()
            pos_list = [int(v) for v in agg.get_column("pos").to_list()]
            neg_list = [int(n - p) for n, p in zip(n_list, pos_list)]
            # If some bins missing (e.g., due to duplicates), coalesce boundaries
            while len(n_list) >= 2 and any(n < self.min_samples_bin for n in n_list):
                # merge the smallest count bin with its smaller chi2 neighbor
                chis = []
                for i in range(len(n_list) - 1):
                    chis.append(self._chi2_pair(pos_list[i], neg_list[i], pos_list[i + 1], neg_list[i + 1]))
                i_min = int(min(range(len(chis)), key=lambda i: chis[i]))
                # merge i_min and i_min+1
                pos_list[i_min] += pos_list[i_min + 1]
                neg_list[i_min] += neg_list[i_min + 1]
                n_list[i_min] += n_list[i_min + 1]
                del pos_list[i_min + 1], neg_list[i_min + 1], n_list[i_min + 1]
                # remove boundary at i_min+1 in edges
                del edges[i_min + 1]
            # main merge loop
            while True:
                if len(n_list) <= self.max_bins:
                    # check chi2 threshold
                    chis = []
                    for i in range(len(n_list) - 1):
                        chis.append(self._chi2_pair(pos_list[i], neg_list[i], pos_list[i + 1], neg_list[i + 1]))
                    if not chis or min(chis) >= crit:
                        break
                    i_min = int(min(range(len(chis)), key=lambda i: chis[i]))
                else:
                    # force merge smallest chi2 until max_bins reached
                    chis = []
                    for i in range(len(n_list) - 1):
                        chis.append(self._chi2_pair(pos_list[i], neg_list[i], pos_list[i + 1], neg_list[i + 1]))
                    i_min = int(min(range(len(chis)), key=lambda i: chis[i]))
                # merge i_min and i_min+1
                pos_list[i_min] += pos_list[i_min + 1]
                neg_list[i_min] += neg_list[i_min + 1]
                n_list[i_min] += n_list[i_min + 1]
                del pos_list[i_min + 1], neg_list[i_min + 1], n_list[i_min + 1]
                del edges[i_min + 1]
                # enforce min_samples_bin
                if n_list[i_min] < self.min_samples_bin and len(n_list) > 1:
                    j = max(0, min(i_min, len(n_list) - 2))
                    pos_list[j] += pos_list[j + 1]
                    neg_list[j] += neg_list[j + 1]
                    n_list[j] += n_list[j + 1]
                    del pos_list[j + 1], neg_list[j + 1], n_list[j + 1]
                    del edges[j + 1]
            self.bins_[col] = _make_bins_safe(edges)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col, edges in self.bins_.items():
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class IsotonicBinningDiscretizer(Transformer):
    """Monotonic supervised binning via isotonic regression.

    Fits an isotonic regression of target vs feature and creates bin boundaries
    at changes in the fitted step function, with optional min bin size and cap on bins.
    """

    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        monotonic: str = "auto",  # 'auto' | 'increasing' | 'decreasing'
        min_samples_bin: int = 20,
        max_bins: Optional[int] = None,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__iso",
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.monotonic = monotonic
        self.min_samples_bin = int(min_samples_bin)
        self.max_bins = max_bins
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}

    def fit(self, df: pl.DataFrame) -> "IsotonicBinningDiscretizer":
        try:
            import numpy as np
            from sklearn.isotonic import IsotonicRegression
        except Exception as e:
            raise ImportError("IsotonicBinningDiscretizer requires scikit-learn and numpy") from e
        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        y_all = df.get_column(self.target).cast(pl.Float64).to_numpy()
        for col in cols:
            x = df.get_column(col).cast(pl.Float64).to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y_all))
            x = x[mask]
            y = y_all[mask]
            if x.size == 0:
                self.bins_[col] = [-float("inf"), float("inf")]
                continue
            # decide monotonic direction
            inc = True
            if self.monotonic == "auto":
                # heuristic: correlation sign
                if np.std(x) > 0 and np.std(y) > 0:
                    corr = np.corrcoef(x, y)[0, 1]
                    inc = bool(corr >= 0)
                else:
                    inc = True
            elif self.monotonic == "decreasing":
                inc = False
            # fit isotonic
            iso = IsotonicRegression(increasing=inc)
            yhat = iso.fit_transform(x, y)
            order = np.argsort(x)
            xs = x[order]
            yh = yhat[order]
            # find change points in yh
            change_idx = [i for i in range(1, len(yh)) if yh[i] != yh[i - 1]]
            # create thresholds at midpoints
            thresholds = []
            for i in change_idx:
                thr = float((xs[i - 1] + xs[i]) / 2.0)
                thresholds.append(thr)
            # enforce min_samples_bin by pruning nearby thresholds
            if thresholds:
                # convert to segment sizes
                seg_starts = [0] + change_idx
                seg_ends = change_idx + [len(xs)]
                seg_sizes = [e - s for s, e in zip(seg_starts, seg_ends)]
                # merge segments smaller than min_samples_bin
                i = 0
                while i < len(seg_sizes):
                    if seg_sizes[i] < self.min_samples_bin and len(seg_sizes) > 1:
                        # merge with neighbor (left if exists else right)
                        if i > 0:
                            seg_sizes[i - 1] += seg_sizes[i]
                            del seg_sizes[i]
                            del thresholds[i - 1]
                            i = max(i - 1, 0)
                        else:
                            seg_sizes[i + 1] += seg_sizes[i]
                            del seg_sizes[i]
                            del thresholds[i]
                    else:
                        i += 1
            # cap number of bins
            if self.max_bins is not None and thresholds:
                max_thr = max(0, self.max_bins - 1)
                if len(thresholds) > max_thr:
                    # keep evenly spaced thresholds
                    idxs = np.linspace(0, len(thresholds) - 1, num=max_thr, dtype=int)
                    thresholds = [thresholds[i] for i in idxs]
            edges = [-float("inf")] + sorted(set(thresholds)) + [float("inf")]
            self.bins_[col] = _make_bins_safe(edges)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col, edges in self.bins_.items():
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class MonotonicOptimalBinningDiscretizer(Transformer):
    """Practical monotonic supervised binning via PAVA on initial quantile bins.

    - Builds many initial quantile bins, computes target rates per bin.
    - Applies Pool-Adjacent-Violators Algorithm (PAVA) to enforce monotonicity.
    - Merges further to satisfy min bin size and optional max_bins.
    - Produces stable bin edges usable with `pl.cut`.
    """

    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        monotonic: str = "auto",  # 'auto' | 'increasing' | 'decreasing'
        initial_bins: int = 50,
        min_bin_size: float | int = 0.02,  # proportion or count
        max_bins: Optional[int] = 10,
        drop_original: bool = True,
        labels_as_int: bool = True,
        suffix: str = "__mob",
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.monotonic = monotonic
        self.initial_bins = int(initial_bins)
        self.min_bin_size = min_bin_size
        self.max_bins = max_bins
        self.drop_original = drop_original
        self.labels_as_int = labels_as_int
        self.suffix = suffix
        self.bins_: Dict[str, List[float]] = {}

    def _pava(self, sums: List[float], cnts: List[int], inc: bool) -> List[tuple[int, int, float, int]]:
        # Returns list of groups: (start_idx, end_idx, sum, cnt)
        groups: List[tuple[int, int, float, int]] = [(i, i, float(sums[i]), int(cnts[i])) for i in range(len(cnts))]
        def rate(g):
            return g[2] / g[3] if g[3] > 0 else 0.0
        i = 0
        while i < len(groups) - 1:
            ok = rate(groups[i]) <= rate(groups[i + 1]) if inc else rate(groups[i]) >= rate(groups[i + 1])
            if ok:
                i += 1
            else:
                # merge i and i+1
                s = groups[i][2] + groups[i + 1][2]
                c = groups[i][3] + groups[i + 1][3]
                start = groups[i][0]
                end = groups[i + 1][1]
                groups[i] = (start, end, s, c)
                del groups[i + 1]
                # move back one step to re-check
                if i > 0:
                    i -= 1
        return groups

    def _enforce_min_size(self, groups: List[tuple[int, int, float, int]], min_cnt: int) -> List[tuple[int, int, float, int]]:
        # Merge groups smaller than min_cnt with neighbor minimizing rate difference
        def rate(g):
            return g[2] / g[3] if g[3] > 0 else 0.0
        i = 0
        while i < len(groups):
            if groups[i][3] < min_cnt and len(groups) > 1:
                # choose neighbor
                left_diff = right_diff = float("inf")
                r_i = rate(groups[i])
                if i > 0:
                    left_diff = abs(r_i - rate(groups[i - 1]))
                if i < len(groups) - 1:
                    right_diff = abs(r_i - rate(groups[i + 1]))
                j = i - 1 if left_diff <= right_diff and i > 0 else i + 1
                j = max(0, min(j, len(groups) - 1))
                # merge i with j (ensure i < j)
                a, b = (i, j) if i < j else (j, i)
                s = groups[a][2] + groups[b][2]
                c = groups[a][3] + groups[b][3]
                start = groups[a][0]
                end = groups[b][1]
                groups[a] = (start, end, s, c)
                del groups[b]
                i = max(a - 1, 0)
            else:
                i += 1
        return groups

    def _cap_max_bins(self, groups: List[tuple[int, int, float, int]], max_bins: int) -> List[tuple[int, int, float, int]]:
        # Merge adjacent pair with smallest rate difference until len <= max_bins
        def rate(g):
            return g[2] / g[3] if g[3] > 0 else 0.0
        while len(groups) > max_bins:
            diffs = [abs(rate(groups[i]) - rate(groups[i + 1])) for i in range(len(groups) - 1)]
            i_min = int(min(range(len(diffs)), key=lambda i: diffs[i]))
            s = groups[i_min][2] + groups[i_min + 1][2]
            c = groups[i_min][3] + groups[i_min + 1][3]
            start = groups[i_min][0]
            end = groups[i_min + 1][1]
            groups[i_min] = (start, end, s, c)
            del groups[i_min + 1]
        return groups

    def fit(self, df: pl.DataFrame) -> "MonotonicOptimalBinningDiscretizer":
        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        cols = _infer_numeric_columns(df, self.columns)
        self.feature_names_in_ = cols
        y = df.get_column(self.target).cast(pl.Float64)
        n_total = df.height
        min_cnt = int(self.min_bin_size * n_total) if isinstance(self.min_bin_size, float) and 0 < self.min_bin_size < 1 else int(self.min_bin_size)
        min_cnt = max(1, min_cnt)
        for col in cols:
            s = df.get_column(col).cast(pl.Float64)
            # initial quantile edges
            n_init = max(2, min(self.initial_bins, int(s.drop_nulls().n_unique())))
            qs = [i / n_init for i in range(1, n_init)]
            quantiles = df.select([pl.col(col).quantile(q) for q in qs]).row(0)
            edges = [-float("inf")] + [float(q) for q in quantiles] + [float("inf")]
            tmp = df.select([
                pl.col(col).cast(pl.Float64).cut(edges, labels=list(range(len(edges) - 1))).alias("__bin__"),
                y.alias("__y__"),
            ])
            agg = tmp.group_by("__bin__").agg([
                pl.len().alias("n"),
                pl.col("__y__").sum().alias("sum"),
            ]).sort("__bin__")
            cnts = [int(v) for v in agg.get_column("n").to_list()]
            sums = [float(v) for v in agg.get_column("sum").to_list()]
            # Determine direction
            inc = True
            if self.monotonic == "auto":
                # simple correlation between s and y
                # compute midpoints for each bin as proxy for x
                mids: List[float] = []
                for i in range(len(edges) - 1):
                    l = edges[i]
                    r = edges[i + 1]
                    if math.isinf(l) and math.isinf(r):
                        mids.append(0.0)
                    elif math.isinf(l):
                        mids.append(r)
                    elif math.isinf(r):
                        mids.append(l)
                    else:
                        mids.append((l + r) / 2.0)
                # weighted correlation sign heuristic
                import numpy as np
                x = np.array(mids)
                y_rate = np.array([sums[i] / cnts[i] if cnts[i] > 0 else 0.0 for i in range(len(cnts))])
                if np.std(x) > 0 and np.std(y_rate) > 0:
                    inc = bool(np.corrcoef(x, y_rate)[0, 1] >= 0)
                else:
                    inc = True
            elif self.monotonic == "decreasing":
                inc = False
            # PAVA
            groups = self._pava(sums, cnts, inc)
            # min size
            groups = self._enforce_min_size(groups, min_cnt)
            # cap bins
            if self.max_bins is not None and self.max_bins > 0:
                groups = self._cap_max_bins(groups, self.max_bins)
            # build final edges
            final_edges = [-float("inf")]
            for g in groups[:-1]:
                # boundary is right edge of this group
                final_edges.append(edges[g[1] + 1])
            final_edges.append(float("inf"))
            self.bins_[col] = _make_bins_safe(final_edges)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col, edges in self.bins_.items():
            labels = list(range(len(edges) - 1)) if self.labels_as_int else None
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Float64).cut(edges, labels=labels).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out
