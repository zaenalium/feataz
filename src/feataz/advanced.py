from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
import polars as pl

from .base import Transformer, _ensure_polars_df


class CrossFitTransformer(Transformer):
    """Leakage-safe cross-fitted wrapper for target-aware transformers.

    It creates K folds, fits a fresh base transformer on K-1 folds, and transforms
    the held-out fold to produce out-of-fold (OOF) features. Then refits a final
    transformer on the full data for use in `transform` on new data.

    Parameters
    - factory: a callable that returns a new, unfitted transformer instance
    - n_splits: number of folds
    - shuffle: shuffle before splitting
    - random_state: seed for shuffling
    - group_column: optional column for group-aware splitting (keeps groups intact)
    """

    def __init__(
        self,
        factory: Callable[[], Transformer],
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
        group_column: Optional[str] = None,
    ) -> None:
        self.factory = factory
        self.n_splits = int(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state
        self.group_column = group_column

        self.fold_models_: List[Transformer] = []
        self.full_model_: Optional[Transformer] = None
        self.oof_columns_: List[str] = []

    def _make_folds(self, df: pl.DataFrame) -> List[pl.Series]:
        n = df.height
        if self.group_column and self.group_column in df.columns:
            groups = df.get_column(self.group_column).to_list()
            # assign groups to folds
            uniq = {}
            for g in groups:
                uniq.setdefault(g, len(uniq))
            grp_ids = np.array([uniq[g] for g in groups])
            rng = np.random.default_rng(self.random_state)
            perm = rng.permutation(len(uniq)) if self.shuffle else np.arange(len(uniq))
            fold_id = np.empty(len(uniq), dtype=int)
            for i, gid in enumerate(perm):
                fold_id[gid] = i % self.n_splits
            return [pl.Series((fold_id[grp_ids] == k)) for k in range(self.n_splits)]
        else:
            idx = np.arange(n)
            rng = np.random.default_rng(self.random_state)
            if self.shuffle:
                rng.shuffle(idx)
            folds = np.array_split(idx, self.n_splits)
            mask_series = []
            inv = np.empty(n, dtype=int)
            for k, f in enumerate(folds):
                inv[...] = 0
                inv[f] = 1
                mask_series.append(pl.Series(inv == 1))
            return mask_series

    def fit(self, df: pl.DataFrame) -> "CrossFitTransformer":
        df = _ensure_polars_df(df)
        masks = self._make_folds(df)
        oof = df
        self.fold_models_.clear()
        self.oof_columns_.clear()
        oof_cols_added: List[str] = []
        for k, m in enumerate(masks):
            train = df.filter(~m)
            valid = df.filter(m)
            model = self.factory()
            model.fit(train)
            self.fold_models_.append(model)
            enc_valid = model.transform(valid)
            # determine new columns by diff
            new_cols = [c for c in enc_valid.columns if c not in valid.columns]
            if not new_cols:
                raise ValueError("Base transformer did not produce new columns")
            oof_cols_added = new_cols
            # attach fold predictions back in order
            enc_valid = enc_valid.with_row_index("__row__")
            valid = valid.with_row_index("__row__")
            merged = valid.select(["__row__"]).join(enc_valid.select(["__row__"] + new_cols), on="__row__", how="left").drop("__row__")
            # Fill into oof for rows of this fold
            oof = oof.with_row_index("__row__")
            oof = oof.join(merged.with_row_index("__row__"), on="__row__", how="left", suffix=f"__f{k}")
            oof = oof.drop("__row__")
        # consolidate oof columns (if duplicates from multiple folds due to suffix)
        # Use first non-null across fold columns for each feature
        for base in oof_cols_added:
            fold_cols = [c for c in oof.columns if c == base or c.startswith(base + "__f")]
            if len(fold_cols) > 1:
                oof = oof.with_columns(pl.coalesce([pl.col(c) for c in fold_cols]).alias(base))
                # drop suffixed ones
                drop_cols = [c for c in fold_cols if c != base]
                oof = oof.drop(drop_cols)
        self.oof_columns_ = oof_cols_added
        # fit full model for transform on new data
        self.full_model_ = self.factory().fit(df)
        self.feature_names_in_ = list(df.columns)
        self.is_fitted_ = True
        self._oof_train_ = oof  # store OOF for reference
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted_ or self.full_model_ is None:
            raise RuntimeError("Call fit before transform")
        return self.full_model_.transform(df)

