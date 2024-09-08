from typing import Any, List, Union, Dict
import polars as pl
import warnings
from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feataz.utils import (
    check_all_variables,
    check_categorical_variables,
    find_all_variables,
    find_categorical_variables,
)

class GetFeatureNamesOutMixin:
    def get_feature_names_out(
        self,
        input_features: Union[List[Union[str, int]], ArrayLike] = None,
    ) -> List[Union[str, int]]:
        """
        Get output feature names for transformation. In other words, returns the
        variable names of transformed dataframe.

        Parameters
        ----------
        input_features : array or list, default=None
            This parameter exits only for compatibility with the Scikit-learn pipeline.

            - If `None`, then `feature_names_in_` is used as feature names in.
            - If an array or list, then `input_features` must match `feature_names_in_`.

        Returns
        -------
        feature_names_out: list
            Transformed feature names.
        """
        check_is_fitted(self)

        if input_features is not None:
            # If input to fit is an array, then the variable names in
            # feature_names_in_ are "x0", "x1","x2" ..."xn".
            if self.feature_names_in_ == [f"x{i}" for i in range(self.n_features_in_)]:

                # If the input was an array, we let the user enter the variable names.
                if len(input_features) == self.n_features_in_:
                    if isinstance(input_features, list):
                        feature_names = input_features
                    else:
                        feature_names = list(input_features)

                    # For transformers that add features to the data.
                    feature_names = self._add_new_feature_names(feature_names)

                    # For transformers that remove features from data, i..e, selectors.
                    feature_names = self._remove_feature_names(
                        feature_names, indices=True
                    )

                    return feature_names

                else:
                    raise ValueError(
                        "The number of input_features does not match the number of "
                        "features seen in the dataframe used in fit."
                    )
            else:
                msg = "input_features is not equal to feature_names_in_"
                if isinstance(input_features, list):
                    if input_features != self.feature_names_in_:
                        raise ValueError(msg)
                elif isinstance(input_features, ndarray):
                    if list(input_features) != self.feature_names_in_:
                        raise ValueError(msg)
                else:
                    raise ValueError(
                        "input_features must be a list or an array. "
                        "Got {input_features} instead."
                    )

        feature_names = self.feature_names_in_

        # For transformers that add features to the dataframe:
        feature_names = self._add_new_feature_names(feature_names)

        # For transformers that remove features from data, i..e, selectors.
        feature_names = self._remove_feature_names(feature_names, indices=False)

        return feature_names

    def _add_new_feature_names(self, feature_names):
        # For transformers that add features to the dataframe:
        if hasattr(self, "_get_new_features_name") and callable(
            self._get_new_features_name
        ):
            feature_names = feature_names + self._get_new_features_name()

            if self.drop_original is True and self.variables_ is not None:
                # Remove names of variables to drop.
                feature_names = [f for f in feature_names if f not in self.variables_]

        return feature_names

    def _remove_feature_names(self, feature_names, indices=False) -> List:
        # For transformers that remove features from data, i..e, selectors.
        if hasattr(self, "features_to_drop_"):
            if indices is True:
                mask = self.get_support(indices=True)
                feature_names = [feature_names[i] for i in mask]
            else:
                feature_names = [
                    f for f in feature_names if f not in self.features_to_drop_
                ]
        return feature_names


class CategoricalInitMixin:
    """Shared initialization parameters across transformers. Sets and checks init
    parameters.

    Parameters
    ----------
    {variables}.

    {ignore_format}
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
    ) -> None:

        if not isinstance(ignore_format, bool):
            raise ValueError(
                "ignore_format takes only booleans True and False. "
                f"Got {ignore_format} instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.ignore_format = ignore_format


class CategoricalInitMixinNA:
    """Shared initialization parameters across transformers. Sets and checks init
    parameters.

    Parameters
    ----------
    {variables}.

    {missing_values}

    {ignore_format}
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
        ignore_format: bool = False,
    ) -> None:

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if not isinstance(ignore_format, bool):
            raise ValueError(
                "ignore_format takes only booleans True and False. "
                f"Got {ignore_format} instead."
            )

        self.variables = _check_variables_input_value(variables)
        self.ignore_format = ignore_format
        self.missing_values = missing_values


class CategoricalMethodsMixin(BaseEstimator, TransformerMixin, GetFeatureNamesOutMixin):
    """Shared methods across categorical transformers.

    - BaseEstimator brings methods get_params() and set_params().
    - TransformerMixin brings method fit_transform()
    - GetFeatureNamesOutMixin brings method get_feature_names_out().
    """

    def _check_na(self, X: pl.DataFrameFrame, variables):
        if self.missing_values == "raise":
            _check_optional_contains_na(X, variables)

    def _check_or_select_variables(self, X: pl.DataFrameFrame):
        """
        Finds categorical variables, or alternatively checks that the variables
        entered by the user are of type object (categorical).
        Checks absence of NA.

        Parameters
        ----------
        X: Pandas DataFrame

        Raises
        ------
        TypeError
            If any user provided variable is not categorical
        ValueError
            If there are no categorical variables in the df or the df is empty
            If the variable(s) contain null values
        """
        # select variables to encode
        if self.ignore_format is True:
            if self.variables is None:
                variables_ = find_all_variables(X)
            else:
                variables_ = check_all_variables(X, self.variables)
        else:
            if self.variables is None:
                variables_ = find_categorical_variables(X)
            else:
                variables_ = check_categorical_variables(X, self.variables)

        return variables_

    def _get_feature_names_in(self, X: pl.DataFrameFrame):
        """
        Returns attributes `featrure_names_in_` and `n_feature_names_in_`, which are
        standard for all transformers in the library.
        """
        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

    def _check_transform_input_and_state(self, X: pl.DataFrameFrame) -> pl.DataFrameFrame:
        """
        Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X: Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values.
            - If the df has different number of features than the df used in fit()

        Returns
        -------
        X: Pandas DataFrame
            The same dataframe entered by the user.
        """

        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check input data contains same number of columns as df used to fit
        _check_X_matches_training_df(X, self.n_features_in_)

        # reorder df to match train set
        X = X[self.feature_names_in_]

        return X

    def transform(self, X: pl.DataFrameFrame) -> pl.DataFrameFrame:
        """Replace categories with the learned parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The dataset to transform.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing the categories replaced by numbers.
        """

        X = self._check_transform_input_and_state(X)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_optional_contains_na(X, self.variables_)

        X = self._encode(X)

        return X

    def _encode(self, X: pl.DataFrameFrame) -> pl.DataFrameFrame:
        # replace categories by the learned parameters
        for feature in self.encoder_dict_.keys():
            X[feature] = X[feature].map(self.encoder_dict_[feature])

            # if original variables are cast as categorical, they will remain
            # categorical after the encoding, and this is probably not desired
            if X[feature].dtype.name == "category":
                if all(isinstance(x, int) for x in X[feature]):
                    X[feature] = X[feature].astype("int")
                else:
                    X[feature] = X[feature].astype("float")

        if self.unseen == "encode":
            X[self.variables_] = X[self.variables_].fillna(self._unseen)
        else:
            # check if nan values were introduced by the transformation
            self._check_nan_values_after_transformation(X)

        return X

    def _check_nan_values_after_transformation(self, X):

        # check if NaN values were introduced by the encoding
        if X[self.variables_].isnull().sum().sum() > 0:

            # obtain the name(s) of the columns have null values
            nan_columns = (
                X[self.encoder_dict_.keys()]
                .columns[X[self.encoder_dict_.keys()].isnull().any()]
                .tolist()
            )

            if len(nan_columns) > 1:
                nan_columns_str = ", ".join(nan_columns)
            else:
                nan_columns_str = nan_columns[0]

            if self.unseen == "ignore":
                warnings.warn(
                    "During the encoding, NaN values were introduced in the feature(s) "
                    f"{nan_columns_str}."
                )
            elif self.unseen == "raise":
                raise ValueError(
                    "During the encoding, NaN values were introduced in the feature(s) "
                    f"{nan_columns_str}."
                )

    def inverse_transform(self, X: pl.DataFrameFrame) -> pl.DataFrameFrame:
        """Convert the encoded variable back to the original values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The transformed dataframe.

        Returns
        -------
        X_tr: pandas dataframe of shape = [n_samples, n_features].
            The un-transformed dataframe, with the categorical variables containing the
            original values.
        """

        X = self._check_transform_input_and_state(X)

        # replace encoded categories by the original values
        for feature in self.encoder_dict_.keys():
            inv_map = {v: k for k, v in self.encoder_dict_[feature].items()}
            X[feature] = X[feature].map(inv_map)

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "categorical"
        # the below test will fail because sklearn requires to check for inf, but
        # you can't check inf of categorical data, numpy returns and error.
        # so we need to leave without this test
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict
