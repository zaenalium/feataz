from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Set, TypeVar, Union, overload

import polars as pl
import polars.selectors as cs

logger = logging.getLogger(__name__)

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)
DataFrameOrLazy = Union[pl.DataFrame, pl.LazyFrame]


def _resolve_column_collisions(
    df: pl.DataFrame,
    new_columns: Sequence[str],
    prefix: str = "",
    suffix: str = "",
) -> List[str]:
    existing = set(df.columns)
    result: List[str] = []
    for col in new_columns:
        candidate = f"{prefix}{col}{suffix}"
        if candidate in existing:
            logger.warning(f"Column '{candidate}' already exists, renaming to avoid collision")
            i = 1
            while f"{candidate}_{i}" in existing:
                i += 1
            candidate = f"{candidate}_{i}"
        result.append(candidate)
    return result


def _ensure_polars_df(df: pl.DataFrame) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df

    pd = None
    if df.__class__.__module__.startswith("pandas"):
        try:
            import pandas as pd
        except ImportError as exc:
            raise TypeError(
                "Pandas support requires installing pandas; install pandas to pass pandas.DataFrame"
            ) from exc
    if pd is not None and isinstance(df, pd.DataFrame):
        try:
            return pl.from_pandas(df)
        except (ImportError, ModuleNotFoundError):
            data = {col: df[col].tolist() for col in df.columns}
            return pl.DataFrame(data)

    raise TypeError("Expected a polars.DataFrame or pandas.DataFrame")


def _ensure_polars_frame(df: DataFrameOrLazy) -> DataFrameOrLazy:
    """Accept DataFrame or LazyFrame, return as-is. Converts pandas if needed."""
    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return df

    pd = None
    if df.__class__.__module__.startswith("pandas"):
        try:
            import pandas as pd
        except ImportError as exc:
            raise TypeError(
                "Pandas support requires installing pandas; install pandas to pass pandas.DataFrame"
            ) from exc
    if pd is not None and isinstance(df, pd.DataFrame):
        try:
            return pl.from_pandas(df)
        except (ImportError, ModuleNotFoundError):
            data = {col: df[col].tolist() for col in df.columns}
            return pl.DataFrame(data)

    raise TypeError("Expected a polars.DataFrame, polars.LazyFrame, or pandas.DataFrame")


def _ensure_eager(df: DataFrameOrLazy) -> pl.DataFrame:
    """Collect LazyFrame to DataFrame if needed."""
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return df


def _get_column_names(df: DataFrameOrLazy) -> List[str]:
    """Get column names from DataFrame or LazyFrame."""
    return df.collect_schema().names()


def _get_column_dtypes(df: DataFrameOrLazy) -> List[pl.DataType]:
    """Get column dtypes from DataFrame or LazyFrame."""
    return df.collect_schema().dtypes()


def _ensure_polars_series(series: pl.Series) -> pl.Series:
    if isinstance(series, pl.Series):
        return series

    pd = None
    if series.__class__.__module__.startswith("pandas"):
        try:
            import pandas as pd
        except ImportError as exc:
            raise TypeError(
                "Pandas support requires installing pandas; install pandas to pass pandas.Series"
            ) from exc
    if pd is not None and isinstance(series, pd.Series):
        name = series.name if series.name is not None else "column_0"
        try:
            return pl.from_pandas(series.to_frame(name=name)).get_column(name)
        except (ImportError, ModuleNotFoundError):
            return pl.Series(name, series.tolist())

    raise TypeError("Expected a polars.Series or pandas.Series")


def _as_list(x: Optional[Sequence[str]], fallback: List[str]) -> List[str]:
    if x is None:
        return list(fallback)
    return list(x)


class ColumnInferenceMixin:
    """Mixin providing utilities for inferring column types from DataFrames."""

    @staticmethod
    def infer_numeric_columns(
        df: DataFrameOrLazy,
        columns: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        include_bool: bool = True,
    ) -> List[str]:
        col_names = _get_column_names(df)
        if columns is not None:
            missing = [c for c in columns if c not in col_names]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            return list(columns)

        exclude_set = set(exclude) if exclude is not None else set()
        selector = cs.numeric()
        if not include_bool:
            selector = selector & ~cs.by_dtype(pl.Boolean)
        result = [c for c in df.select(selector).collect_schema().names() if c not in exclude_set]
        return result

    @staticmethod
    def infer_categorical_columns(
        df: DataFrameOrLazy,
        columns: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> List[str]:
        col_names = _get_column_names(df)
        if columns is not None:
            missing = [c for c in columns if c not in col_names]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            return list(columns)

        exclude_set = set(exclude) if exclude is not None else set()
        selector = cs.by_dtype(pl.String, pl.Categorical)
        result = [c for c in df.select(selector).collect_schema().names() if c not in exclude_set]
        return result

    @staticmethod
    def infer_datetime_columns(
        df: DataFrameOrLazy,
        columns: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> List[str]:
        col_names = _get_column_names(df)
        if columns is not None:
            missing = [c for c in columns if c not in col_names]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            return list(columns)

        exclude_set = set(exclude) if exclude is not None else set()
        selector = cs.by_dtype(pl.Date, pl.Datetime)
        result = [c for c in df.select(selector).collect_schema().names() if c not in exclude_set]
        return result


class ValidationMixin:
    """Mixin providing validation utilities for transformers."""

    @staticmethod
    def validate_fitted(is_fitted: bool, class_name: str) -> None:
        """Validate that the transformer is fitted.

        Parameters
        ----------
        is_fitted : bool
            Whether the transformer is fitted.
        class_name : str
            Name of the transformer class for error message.

        Raises
        ------
        RuntimeError
            If the transformer is not fitted.
        """
        if not is_fitted:
            raise RuntimeError(f"{class_name} is not fitted. Call fit() first.")

    @staticmethod
    def validate_columns_exist(df: DataFrameOrLazy, columns: Sequence[str]) -> None:
        """Validate that all columns exist in the DataFrame.

        Parameters
        ----------
        df : DataFrameOrLazy
            Input DataFrame or LazyFrame.
        columns : Sequence[str]
            Columns to validate.

        Raises
        ------
        ValueError
            If any columns are not found in the DataFrame.
        """
        col_names = _get_column_names(df)
        missing = [c for c in columns if c not in col_names]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

    @staticmethod
    def validate_non_empty(df: DataFrameOrLazy) -> None:
        """Validate that the DataFrame is not empty.

        Parameters
        ----------
        df : DataFrameOrLazy
            Input DataFrame or LazyFrame.

        Raises
        ------
        ValueError
            If the DataFrame is empty.
        """
        if isinstance(df, pl.LazyFrame):
            if df.select(pl.len()).collect().item() == 0:
                raise ValueError("DataFrame is empty.")
        elif df.is_empty():
            raise ValueError("DataFrame is empty.")

    @staticmethod
    def validate_target_column(df: DataFrameOrLazy, target: str) -> None:
        """Validate that the target column exists.

        Parameters
        ----------
        df : DataFrameOrLazy
            Input DataFrame or LazyFrame.
        target : str
            Target column name.

        Raises
        ------
        ValueError
            If the target column is not found.
        """
        col_names = _get_column_names(df)
        if target not in col_names:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")


class Transformer(ABC, ColumnInferenceMixin, ValidationMixin):
    """Base class for all transformers with sklearn-style API.

    Subclasses must implement:
        - fit(self, df: pl.DataFrame) -> "Transformer"
        - transform(self, df: DataFrameOrLazy) -> DataFrameOrLazy

    Optional methods to implement:
        - inverse_transform(self, df: DataFrameOrLazy) -> DataFrameOrLazy

    Attributes
    ----------
    feature_names_in_ : List[str] | None
        Names of columns seen during fit.
    feature_names_out_ : List[str]
        Names of output columns after transform.
    is_fitted_ : bool
        Whether the transformer has been fitted.
    supports_lazy_ : bool
        Whether this transformer supports LazyFrame in transform. Default True.
    """

    feature_names_in_: List[str] | None = None
    feature_names_out_: List[str] = []
    is_fitted_: bool = False
    supports_lazy_: bool = True
    _output_format: str = "default"

    @abstractmethod
    def fit(self, df: pl.DataFrame) -> "Transformer":
        """Fit the transformer to the data.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame.

        Returns
        -------
        Transformer
            Fitted transformer.
        """
        pass

    @abstractmethod
    def transform(self, df: DataFrameOrLazy) -> DataFrameOrLazy:
        """Transform the data.

        Parameters
        ----------
        df : DataFrameOrLazy
            Input DataFrame or LazyFrame.

        Returns
        -------
        DataFrameOrLazy
            Transformed DataFrame or LazyFrame.
        """
        pass

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform the data in one step.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: DataFrameOrLazy) -> DataFrameOrLazy:
        """Inverse transform the data (if applicable).

        Parameters
        ----------
        df : DataFrameOrLazy
            Transformed DataFrame or LazyFrame.

        Returns
        -------
        DataFrameOrLazy
            Original DataFrame or LazyFrame.

        Raises
        ------
        NotImplementedError
            If inverse transform is not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform."
        )

    def get_feature_names_out(self) -> List[str]:
        """Get the names of the output columns.

        Returns
        -------
        List[str]
            Names of output columns.
        """
        names = getattr(self, "feature_names_out_", None)
        if names is None:
            return []
        return list(names)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this transformer.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters of sub-objects.

        Returns
        -------
        Dict[str, Any]
            Parameter names and values.
        """
        params: Dict[str, Any] = {}
        sig = inspect.signature(self.__init__)
        for key in sig.parameters:
            if key == "self":
                continue
            value = getattr(self, key, None)
            if deep and hasattr(value, "get_params"):
                value = value.get_params(deep=True)
            params[key] = value
        return params

    def set_params(self, **params: Any) -> "Transformer":
        """Set parameters for this transformer.

        Parameters
        ----------
        **params : Any
            Parameter names and values to set.

        Returns
        -------
        Transformer
            Self for method chaining.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def set_output(self, *, transform: str = "default") -> "Transformer":
        """Set the output format for transform.

        Parameters
        ----------
        transform : str, default="default"
            Output format. Options: "default" (polars), "pandas".

        Returns
        -------
        Transformer
            Self for method chaining.
        """
        if transform not in ("default", "pandas"):
            raise ValueError(f"Invalid output format: {transform}. Use 'default' or 'pandas'.")
        self._output_format = transform
        return self

    def _wrap_output(self, df: pl.DataFrame) -> pl.DataFrame:
        """Wrap output according to configured format.

        Parameters
        ----------
        df : pl.DataFrame
            Output DataFrame.

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            Wrapped output.
        """
        if getattr(self, "_output_format", "default") == "pandas":
            return df.to_pandas()
        return df

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the transformer state to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Serialized state.
        """
        state: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if not k.endswith("_") or k.startswith("_"):
                continue
            tname = type(v).__name__
            if tname in {"DecisionTreeClassifier", "DecisionTreeRegressor", "KMeans", "IsolationForest"}:
                continue
            try:
                pl.Series([v])
                state[k] = v
            except Exception:
                if isinstance(v, pl.DataFrame):
                    state[k] = {"__type__": "pldf", "columns": v.columns, "rows": v.rows()}
                elif isinstance(v, (list, dict, str, int, float, type(None))):
                    state[k] = v
        state["__class__"] = self.__class__.__name__
        return state

    def from_dict(self, state: Dict[str, Any]) -> "Transformer":
        """Restore the transformer state from a dictionary.

        Parameters
        ----------
        state : Dict[str, Any]
            Serialized state.

        Returns
        -------
        Transformer
            Self for method chaining.
        """
        for k, v in state.items():
            if k == "__class__":
                continue
            if isinstance(v, dict) and v.get("__type__") == "pldf":
                df = pl.DataFrame(v["rows"], schema=v["columns"])
                setattr(self, k, df)
            else:
                setattr(self, k, v)
        return self

    def __sklearn_clone__(self) -> "Transformer":
        """Support for sklearn.clone()."""
        params = self.get_params(deep=True)
        return self.__class__(**params)

    def __sklearn_tags__(self) -> Dict[str, Any]:
        """Support for sklearn tags."""
        return {
            "requires_y": False,
            "X_types": ["2darray", "sparse", "dataframe"],
            "preserves_dtype": [],
            "allow_nan": True,
        }
