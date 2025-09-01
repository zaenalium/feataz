from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import polars as pl

from .base import Transformer, _as_list, _ensure_polars_df


def _infer_categorical_columns(df: pl.DataFrame, columns: Optional[Sequence[str]]) -> List[str]:
    if columns is not None:
        return list(columns)
    cats: List[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if pl.datatypes.is_string_dtype(dtype) or dtype == pl.Categorical:
            cats.append(name)
    return cats


class OneHotEncoder(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        drop_first: bool = False,
        drop_original: bool = True,
        prefix_sep: str = "__",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.drop_first = drop_first
        self.drop_original = drop_original
        self.prefix_sep = prefix_sep
        self.categories_: Dict[str, List[str]] = {}

    def fit(self, df: pl.DataFrame) -> "OneHotEncoder":
        df = _ensure_polars_df(df)
        cols = _infer_categorical_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.categories_.clear()
        for col in cols:
            cats = (
                df.get_column(col)
                .drop_nulls()
                .unique()
                .cast(pl.Utf8)
                .to_list()
            )
            cats_sorted = sorted(cats)
            if self.drop_first and len(cats_sorted) > 0:
                cats_sorted = cats_sorted[1:]
            self.categories_[col] = cats_sorted
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        df = _ensure_polars_df(df)
        out = df
        for col in self.feature_names_in_ or []:
            cats = self.categories_[col]
            new_cols = []
            for cat in cats:
                new_name = f"{col}{self.prefix_sep}{cat}"
                expr = (pl.col(col).cast(pl.Utf8) == pl.lit(cat)).cast(pl.Int8).alias(new_name)
                new_cols.append(expr)
            out = out.with_columns(new_cols)
            if self.drop_original:
                out = out.drop(col)
        return out


class OrdinalEncoder(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        handle_unknown: str = "use_encoded_value",
        unknown_value: int = -1,
        drop_original: bool = True,
        suffix: str = "__ord",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.handle_unknown = handle_unknown
        self.unknown_value = int(unknown_value)
        self.drop_original = drop_original
        self.suffix = suffix
        self.mappings_: Dict[str, Dict[str, int]] = {}

    def fit(self, df: pl.DataFrame) -> "OrdinalEncoder":
        df = _ensure_polars_df(df)
        cols = _infer_categorical_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.mappings_.clear()
        for col in cols:
            cats = (
                df.get_column(col)
                .drop_nulls()
                .unique()
                .cast(pl.Utf8)
                .to_list()
            )
            mapping = {cat: i for i, cat in enumerate(sorted(cats))}
            self.mappings_[col] = mapping
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            mapping = self.mappings_[col]
            map_df = pl.DataFrame({col: list(mapping.keys()), f"{col}{self.suffix}": list(mapping.values())})
            out = out.join(map_df, on=col, how="left")
            if self.handle_unknown == "use_encoded_value":
                out = out.with_columns(
                    pl.col(f"{col}{self.suffix}").fill_null(self.unknown_value)
                )
            elif self.handle_unknown == "ignore":
                pass
            else:
                # error
                if out.select(pl.col(f"{col}{self.suffix}").is_null().any()).item():
                    raise ValueError(f"Unknown categories found in column {col}")
            if self.drop_original:
                out = out.drop(col)
        return out


class CountFrequencyEncoder(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        normalize: bool = False,
        drop_original: bool = True,
        suffix: str = "__count",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.normalize = normalize
        self.drop_original = drop_original
        self.suffix = suffix
        self.counts_: Dict[str, pl.DataFrame] = {}

    def fit(self, df: pl.DataFrame) -> "CountFrequencyEncoder":
        df = _ensure_polars_df(df)
        cols = _infer_categorical_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.counts_.clear()
        n = df.height
        for col in cols:
            cnt = df.group_by(col).len().rename({"len": f"{col}{self.suffix}"})
            if self.normalize and n > 0:
                cnt = cnt.with_columns(pl.col(f"{col}{self.suffix}") / pl.lit(float(n)))
            self.counts_[col] = cnt
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            cnt = self.counts_[col]
            out = out.join(cnt, on=col, how="left")
            # unseen categories -> 0
            out = out.with_columns(pl.col(f"{col}{self.suffix}").fill_null(0.0))
            if self.drop_original:
                out = out.drop(col)
        return out


class MeanEncoder(Transformer):
    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        smoothing: float = 0.0,
        drop_original: bool = True,
        suffix: str = "__mean",
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.smoothing = float(smoothing)
        self.drop_original = drop_original
        self.suffix = suffix
        self.global_mean_: float | None = None
        self.encodings_: Dict[str, pl.DataFrame] = {}

    def fit(self, df: pl.DataFrame) -> "MeanEncoder":
        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        cols = _infer_categorical_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.global_mean_ = df.get_column(self.target).mean()
        self.encodings_.clear()
        for col in cols:
            agg = df.group_by(col).agg([
                pl.len().alias("cnt"),
                pl.col(self.target).mean().alias("mean_t"),
            ])
            if self.smoothing > 0:
                m = float(self.global_mean_)
                agg = agg.with_columns(
                    (((pl.col("cnt") * pl.col("mean_t") + self.smoothing * m)) / (pl.col("cnt") + self.smoothing)).alias(
                        f"{col}{self.suffix}"
                    )
                )
            else:
                agg = agg.with_columns(pl.col("mean_t").alias(f"{col}{self.suffix}"))
            self.encodings_[col] = agg.select([col, f"{col}{self.suffix}"])
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        global_mean = float(self.global_mean_ if self.global_mean_ is not None else 0.0)
        for col in self.feature_names_in_ or []:
            enc = self.encodings_[col]
            out = out.join(enc, on=col, how="left")
            out = out.with_columns(pl.col(f"{col}{self.suffix}").fill_null(global_mean))
            if self.drop_original:
                out = out.drop(col)
        return out


class WeightOfEvidenceEncoder(Transformer):
    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        eps: float = 0.5,
        drop_original: bool = True,
        suffix: str = "__woe",
        multi_class: str = "auto",  # 'binary' | 'ovr' | 'auto'
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.eps = float(eps)
        self.drop_original = drop_original
        self.suffix = suffix
        self.multi_class = multi_class
        self.encodings_: Dict[str, pl.DataFrame] = {}

    def fit(self, df: pl.DataFrame) -> "WeightOfEvidenceEncoder":
        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        # check target classes
        unique_vals = df.get_column(self.target).drop_nulls().unique().to_list()
        n_classes = len(unique_vals)
        mode = self.multi_class
        if mode == "auto":
            mode = "binary" if n_classes <= 2 else "ovr"
        if mode not in {"binary", "ovr"}:
            raise ValueError("multi_class must be 'binary' or 'ovr'")
        cols = _infer_categorical_columns(df, self.columns)
        self.feature_names_in_ = cols
        eps = self.eps
        self.encodings_.clear()
        if mode == "binary":
            total_pos = df.filter(pl.col(self.target) == 1).height
            total_neg = df.filter(pl.col(self.target) == 0).height
        else:
            classes = sorted(df.get_column(self.target).drop_nulls().unique().to_list())
            totals = {}
            for c in classes:
                totals[c] = (
                    df.select([
                        (pl.col(self.target) == c).sum().alias("pos"),
                        (pl.col(self.target) != c).sum().alias("neg"),
                    ])
                    .row(0)
                )
        eps = self.eps
        for col in cols:
            if mode == "binary":
                agg = (
                    df.group_by(col)
                    .agg([
                        (pl.col(self.target) == 1).sum().alias("pos"),
                        (pl.col(self.target) == 0).sum().alias("neg"),
                    ])
                    .with_columns([
                        (pl.col("pos") + eps) / (pl.lit(float(total_pos)) + eps * 2).alias("p_pos"),
                        (pl.col("neg") + eps) / (pl.lit(float(total_neg)) + eps * 2).alias("p_neg"),
                    ])
                    .with_columns((pl.col("p_pos") / pl.col("p_neg")).log().alias(f"{col}{self.suffix}"))
                    .select([col, f"{col}{self.suffix}"])
                )
                self.encodings_[col] = agg
            else:
                # one-vs-rest WoE per class
                frames = []
                classes = sorted(df.get_column(self.target).drop_nulls().unique().to_list())
                for c in classes:
                    tot_pos, tot_neg = totals[c]
                    a = (
                        df.group_by(col)
                        .agg([
                            (pl.col(self.target) == c).sum().alias("pos"),
                            (pl.col(self.target) != c).sum().alias("neg"),
                        ])
                        .with_columns([
                            (pl.col("pos") + eps) / (pl.lit(float(tot_pos)) + eps * 2).alias("p_pos"),
                            (pl.col("neg") + eps) / (pl.lit(float(tot_neg)) + eps * 2).alias("p_neg"),
                        ])
                        .with_columns((pl.col("p_pos") / pl.col("p_neg")).log().alias(f"{col}{self.suffix}__{c}"))
                        .select([col, f"{col}{self.suffix}__{c}"])
                    )
                    frames.append(a)
                # merge per class
                enc = frames[0]
                for fr in frames[1:]:
                    enc = enc.join(fr, on=col, how="outer")
                self.encodings_[col] = enc
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            enc = self.encodings_[col]
            out = out.join(enc, on=col, how="left")
            # fill all new columns with 0.0 for unseen categories
            new_cols = [c for c in enc.columns if c != col]
            for nc in new_cols:
                out = out.with_columns(pl.col(nc).fill_null(0.0))
            if self.drop_original:
                out = out.drop(col)
        return out


class DecisionTreeEncoder(Transformer):
    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        problem: str = "auto",  # 'classification' | 'regression' | 'auto'
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        drop_original: bool = True,
        suffix: str = "__dte",
        random_state: Optional[int] = 42,
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.problem = problem
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.drop_original = drop_original
        self.suffix = suffix
        self.random_state = random_state
        self.trees_: Dict[str, object] = {}
        self.ordinal_maps_: Dict[str, Dict[str, int]] = {}
        self.global_value_: float | None = None

    def fit(self, df: pl.DataFrame) -> "DecisionTreeEncoder":
        try:
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        except Exception as e:  # pragma: no cover - optional dep
            raise ImportError(
                "DecisionTreeEncoder requires scikit-learn. Install with `pip install scikit-learn`."
            ) from e

        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        cols = _infer_categorical_columns(df, self.columns)
        self.feature_names_in_ = cols
        y = df.get_column(self.target).to_numpy()
        self.global_value_ = float(pl.Series(y).mean())
        self.trees_.clear()
        self.ordinal_maps_.clear()

        # Determine problem type if auto
        problem = self.problem
        if problem == "auto":
            # heuristic: <= 20 unique -> classification
            n_unique = len(pl.Series(y).drop_nulls().unique())
            if n_unique <= 20 and pl.Series(y).dtype in (pl.Int64, pl.Int32, pl.UInt32, pl.UInt64, pl.Boolean):
                problem = "classification"
            else:
                problem = "regression"

        for col in cols:
            # ordinal map
            cats = (
                df.get_column(col).drop_nulls().cast(pl.Utf8).unique().to_list()
            )
            mapping = {c: i for i, c in enumerate(sorted(cats))}
            self.ordinal_maps_[col] = mapping
            X = (
                df.select(pl.col(col).cast(pl.Utf8))
                .with_columns(pl.col(col).map_elements(lambda x: mapping.get(x, -1)).alias(col))
                .get_column(col)
                .to_numpy()
                .reshape(-1, 1)
            )
            if problem == "classification":
                tree = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
                tree.fit(X, y)
            else:
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
                tree.fit(X, y)
            self.trees_[col] = tree
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        global_val = float(self.global_value_ if self.global_value_ is not None else 0.0)
        for col in self.feature_names_in_ or []:
            mapping = self.ordinal_maps_[col]
            tree = self.trees_[col]
            X = (
                df.select(pl.col(col).cast(pl.Utf8))
                .with_columns(pl.col(col).map_elements(lambda x: mapping.get(x, -1)).alias(col))
                .get_column(col)
                .to_numpy()
                .reshape(-1, 1)
            )
            # predictions as encoding
            try:
                y_hat = tree.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
            except Exception:
                y_hat = tree.predict(X)
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.Series(name=new_col, values=y_hat))
            if self.drop_original:
                out = out.drop(col)
            # Replace possible nans from unknowns
            out = out.with_columns(pl.col(new_col).fill_nan(global_val))
        return out


class RareLabelEncoder(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        min_frequency: float | int = 0.01,
        rare_label: str = "__rare__",
        drop_original: bool = False,
        suffix: str = "__rl",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.min_frequency = min_frequency
        self.rare_label = rare_label
        self.drop_original = drop_original
        self.suffix = suffix
        self.frequent_: Dict[str, set] = {}

    def fit(self, df: pl.DataFrame) -> "RareLabelEncoder":
        df = _ensure_polars_df(df)
        cols = _infer_categorical_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.frequent_.clear()
        n = df.height
        for col in cols:
            vc = df.group_by(col).len()
            if isinstance(self.min_frequency, float) and 0 < self.min_frequency < 1:
                vc = vc.with_columns((pl.col("len") / float(n)).alias("freq"))
                keep = vc.filter(pl.col("freq") >= float(self.min_frequency)).get_column(col).to_list()
            else:
                keep = vc.filter(pl.col("len") >= int(self.min_frequency)).get_column(col).to_list()
            self.frequent_[col] = set(map(str, keep))
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            keep = self.frequent_[col]
            new_col = f"{col}{self.suffix}" if not self.drop_original else col
            out = out.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8).is_in(list(keep)))
                .then(pl.col(col).cast(pl.Utf8))
                .otherwise(pl.lit(self.rare_label))
                .alias(new_col)
            )
            if self.drop_original and new_col == col:
                # replaced in place
                pass
            elif self.drop_original:
                out = out.drop(col)
        return out


class StringSimilarityEncoder(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        anchors: Optional[Dict[str, Sequence[str]]] = None,
        top_k_anchors: int = 5,
        drop_original: bool = True,
        prefix: str = "__sim__",
        backend: str = "auto",  # 'auto' | 'difflib' | 'rapidfuzz'
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.anchors = anchors
        self.top_k_anchors = int(top_k_anchors)
        self.drop_original = drop_original
        self.prefix = prefix
        self.backend = backend
        self.anchors_: Dict[str, List[str]] = {}

    def fit(self, df: pl.DataFrame) -> "StringSimilarityEncoder":
        from collections import Counter

        df = _ensure_polars_df(df)
        cols = _infer_categorical_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.anchors_ = {}
        for col in cols:
            if self.anchors and col in self.anchors:
                anchors = [str(a) for a in self.anchors[col]]
            else:
                # pick top-k frequent categories as anchors
                values = df.get_column(col).drop_nulls().cast(pl.Utf8).to_list()
                common = [v for v, _ in Counter(values).most_common(self.top_k_anchors)]
                anchors = common
            self.anchors_[col] = anchors
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            anchors = self.anchors_[col]
            # compute mapping for unique values in this DF to avoid repeated Python calls
            uniques = (
                df.get_column(col).drop_nulls().cast(pl.Utf8).unique().to_list()
            )
            # build similarity dict per anchor
            sim_maps = {a: {} for a in anchors}
            backend = self.backend
            if backend == "auto":
                try:
                    import rapidfuzz  # type: ignore
                    backend = "rapidfuzz"
                except Exception:
                    backend = "difflib"
            if backend == "rapidfuzz":
                from rapidfuzz import fuzz  # type: ignore
                for u in uniques:
                    for a in anchors:
                        sim_maps[a][u] = float(fuzz.ratio(u, a)) / 100.0
            else:
                import difflib
                for u in uniques:
                    for a in anchors:
                        sim_maps[a][u] = difflib.SequenceMatcher(None, u, a).ratio()
            # generate columns via map_elements using the dicts
            for a in anchors:
                new_col = f"{col}{self.prefix}{a}"
                mapping = sim_maps[a]
                out = out.with_columns(
                    pl.col(col)
                    .cast(pl.Utf8)
                    .map_elements(lambda x, m=mapping: float(m.get(x, 0.0)), return_dtype=pl.Float64)
                    .alias(new_col)
                )
            if self.drop_original:
                out = out.drop(col)
        return out


class HashEncoder(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        n_components: int = 8,
        drop_original: bool = True,
        suffix: str = "__hash",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.n_components = int(n_components)
        self.drop_original = drop_original
        self.suffix = suffix

    def fit(self, df: pl.DataFrame) -> "HashEncoder":
        df = _ensure_polars_df(df)
        self.feature_names_in_ = _infer_categorical_columns(df, self.columns)
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        import hashlib

        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        m = self.n_components
        for col in self.feature_names_in_ or []:
            def hfun(x: str, M=m) -> int:
                if x is None:
                    return 0
                return int(hashlib.md5(str(x).encode("utf-8")).hexdigest(), 16) % M
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.col(col).cast(pl.Utf8).map_elements(hfun, return_dtype=pl.Int32).alias(new_col))
            if self.drop_original:
                out = out.drop(col)
        return out


class BinaryEncoder(Transformer):
    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        drop_original: bool = True,
        prefix: str = "__bin__",
    ) -> None:
        self.columns = None if columns is None else list(columns)
        self.drop_original = drop_original
        self.prefix = prefix
        self.ordinal_maps_: Dict[str, Dict[str, int]] = {}
        self.n_bits_: Dict[str, int] = {}

    def fit(self, df: pl.DataFrame) -> "BinaryEncoder":
        df = _ensure_polars_df(df)
        cols = _infer_categorical_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.ordinal_maps_.clear()
        self.n_bits_.clear()
        for col in cols:
            cats = df.get_column(col).drop_nulls().cast(pl.Utf8).unique().to_list()
            mapping = {c: i + 1 for i, c in enumerate(sorted(cats))}  # reserve 0 for null/unseen
            self.ordinal_maps_[col] = mapping
            n = len(mapping) + 1
            n_bits = max(1, (n - 1).bit_length())
            self.n_bits_[col] = n_bits
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        for col in self.feature_names_in_ or []:
            mapping = self.ordinal_maps_[col]
            n_bits = self.n_bits_[col]
            out = out.with_columns(pl.col(col).cast(pl.Utf8).map_elements(lambda x, m=mapping: m.get(x, 0), return_dtype=pl.Int64).alias(f"{col}{self.prefix}ord"))
            for b in range(n_bits):
                out = out.with_columns(((pl.col(f"{col}{self.prefix}ord") >> b) & 1).cast(pl.Int8).alias(f"{col}{self.prefix}{b}"))
            out = out.drop(f"{col}{self.prefix}ord")
            if self.drop_original:
                out = out.drop(col)
        return out


class LeaveOneOutEncoder(Transformer):
    def __init__(
        self,
        target: str,
        columns: Optional[Sequence[str]] = None,
        smoothing: float = 0.0,
        drop_original: bool = True,
        suffix: str = "__loo",
    ) -> None:
        self.target = target
        self.columns = None if columns is None else list(columns)
        self.smoothing = float(smoothing)
        self.drop_original = drop_original
        self.suffix = suffix
        self.stats_: Dict[str, pl.DataFrame] = {}
        self.global_mean_: float | None = None

    def fit(self, df: pl.DataFrame) -> "LeaveOneOutEncoder":
        df = _ensure_polars_df(df)
        if self.target not in df.columns:
            raise ValueError(f"target column '{self.target}' not found")
        cols = _infer_categorical_columns(df, self.columns)
        self.feature_names_in_ = cols
        self.global_mean_ = float(df.get_column(self.target).mean())
        self.stats_.clear()
        for col in cols:
            agg = df.group_by(col).agg([
                pl.len().alias("cnt"),
                pl.col(self.target).sum().alias("sum_t"),
            ])
            self.stats_[col] = agg
        self.is_fitted_ = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Call fit before transform")
        out = df
        global_mean = float(self.global_mean_ if self.global_mean_ is not None else 0.0)
        for col in self.feature_names_in_ or []:
            stats = self.stats_[col]
            out = out.join(stats, on=col, how="left")
            # (sum - y) / (cnt - 1) with smoothing
            enc = ((pl.col("sum_t") - pl.col(self.target)) / (pl.col("cnt") - 1)).alias("_tmp_loo")
            if self.smoothing > 0:
                m = global_mean
                enc = (((pl.col("cnt") - 1) * enc + self.smoothing * m) / ((pl.col("cnt") - 1) + self.smoothing)).alias("_tmp_loo")
            out = out.with_columns(enc)
            new_col = f"{col}{self.suffix}"
            out = out.with_columns(pl.when(pl.col("cnt") > 1).then(pl.col("_tmp_loo")).otherwise(global_mean).alias(new_col))
            out = out.drop(["cnt", "sum_t", "_tmp_loo"])
            if self.drop_original:
                out = out.drop(col)
        return out
