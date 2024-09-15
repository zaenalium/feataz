import numpy as np
import pandas as pd
import polars as pl
from typing import Any, List, Union, Dict
import polars as pl
import polars as pl
from typing import List, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype as is_numeric

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
)

Variables = Union[int, str, List[Union[str, int]], Dict]
DATETIME_TYPES = ("datetimetz", "datetime")

def _check_variables_input_value(variables: Variables) -> Any:
    """
    Checks that the input value for the `variables` parameter located in the init of
    all Feature-engine transformers is of the correct type.
    Allowed  values are None, int, str or list of strings and integers.

    Parameters
    ----------
    variables : string, int, list of strings, list of integers. Default=None

    Returns
    -------
    variables: same as input
    """

    msg = (
        "`variables` should contain a string, an integer or a list of strings or "
        f"integers. Got {variables} instead."
    )
    msg_dupes = "The list entered in `variables` contains duplicated variable names."
    msg_empty = "The list of `variables` is empty."

    if variables is not None:
        if isinstance(variables, list):
            if not all(isinstance(i, (str, int)) for i in variables):
                raise ValueError(msg)
            if len(variables) == 0:
                raise ValueError(msg_empty)
            if len(variables) != len(set(variables)):
                raise ValueError(msg_dupes)
        else:
            if not isinstance(variables, (str, int)):
                raise ValueError(msg)
    return variables


def find_categorical_variables(X: pl.DataFrame) -> List[Union[str, int]]:
    """
    Returns a list with the names of all the categorical variables in a dataframe.
    Note that variables cast as object that can be parsed to datetime will be
    excluded.

    More details in the :ref:`User Guide <find_cat_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    Returns
    -------
    variables: List
        The names of the categorical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_categorical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = find_categorical_variables(X)
    >>> var_
    ['var_cat']
    """
    variables = [
        column
        for column in X.select_dtypes(include=["O", "category"]).columns
        if _is_categorical_and_is_not_datetime(X[column])
    ]
    if len(variables) == 0:
        raise TypeError(
            "No categorical variables found in this dataframe. Please check "
            "variable format with pandas dtypes."
        )
    return variables


def find_all_variables(
    X: pd.DataFrame,
    exclude_datetime: bool = False,
) -> List[Union[str, int]]:
    """
    Returns a list with the names of all the variables in the dataframe. It has the
    option to exlcude variables that can be parsed as datetime or datetimetz.

    More details in the :ref:`User Guide <find_all_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    exclude_datetime: bool, default=False
        Whether to exclude datetime variables.

    Returns
    -------
    variables: List
        The names of the variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_all_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> vars_all = find_all_variables(X)
    >>> vars_all
    ['var_num', 'var_cat', 'var_date']
    """
    if exclude_datetime is True:
        variables = X.select_dtypes(exclude=DATETIME_TYPES).columns.to_list()
        variables = [
            var
            for var in variables
            if is_numeric(X[var]) or not _is_categorical_and_is_datetime(X[var])
        ]
    else:
        variables = X.columns.to_list()
    return variables


def _check_optional_contains_na(
    X: pl.DataFrame, variables: List[Union[str, int]]
) -> None:
    """
    Checks if DataFrame contains null values in the selected columns.

    Parameters
    ----------
    X : Pandas DataFrame

    variables : List
        The selected group of variables in which null values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain null values.
    """

    if X[variables].isnull().any().any():
        raise ValueError(
            "Some of the variables in the dataset contain NaN. Check and "
            "remove those before using this transformer or set the parameter "
            "`missing_values='ignore'` when initialising this transformer."
        )
    
def check_data(X: Union[np.generic, np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Checks if the input is a DataFrame and then creates a copy. This is an important
    step not to accidentally transform the original dataset entered by the user.

    If the input is a numpy array, it converts it to a pandas Dataframe. The column
    names are strings representing the column index starting at 0.

    Feature-engine was originally designed to work with pandas dataframes. However,
    allowing numpy arrays as input allows 2 things:

    We can use the Scikit-learn tests for transformers provided by the
    `check_estimator` function to test the compatibility of our transformers with
    sklearn functionality.

    Feature-engine transformers can be used within a Scikit-learn Pipeline together
    with Scikit-learn transformers like the `SimpleImputer`, which return by default
    Numpy arrays.

    Parameters
    ----------
    X : pandas Dataframe or numpy array.
        The input to check and copy or transform.

    Raises
    ------
    TypeError
        If the input is not a Pandas DataFrame or a numpy array.
    ValueError
        If the input is an empty dataframe.

    Returns
    -------
    X : polars Dataframe.
        A copy of original DataFrame or a converted Numpy array.
    """
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValueError("Input data contains duplicated variable names.")
        data = pl.from_pandas(X)

    elif isinstance(X, (np.generic, np.ndarray)):
        # If input is scalar raise error
        if X.ndim == 0:
            raise ValueError(
                "Expected 2D array, got scalar array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(X)
            )
        # If input is 1D raise error
        if X.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(X)
            )

        data = pl.DataFrame(X)
        data.columns = [f"x{i}" for i in range(X.shape[1])]
    elif isinstance(X, pl.DataFrame): 
        data = X.clone()
    elif issparse(X):
        raise TypeError("This transformer does not support sparse matrices.")
    else:
        raise TypeError(
            f"X must be a numpy array or pandas dataframe. Got {type(X)} instead."
        )

    if X.__len__() == 0:
        raise ValueError(
            "0 feature(s) (shape=%s) while a minimum of %d is required." % (X.shape, 1)
        )

    return data


# def validate_variables(variables: Variables) -> Any:
#     """
#     Checks that the input value for the `variables` parameter located in the init of
#     all Feature-engine transformers is of the correct type.
#     Allowed  values are None, int, str or list of strings and integers.

#     Parameters
#     ----------
#     variables : string, int, list of strings, list of integers. Default=None

#     Returns
#     -------
#     variables: same as input
#     """

#     msg = (
#         "`variables` should contain a string, an integer or a list of strings or "
#         f"integers. Got {variables} instead."
#     )
#     msg_dupes = "The list entered in `variables` contains duplicated variable names."
#     msg_empty = "The list of `variables` is empty."

#     if variables is not None:
#         if isinstance(variables, list):
#             if not all(isinstance(i, (str, int)) for i in variables):
#                 raise ValueError(msg)
#             if len(variables) == 0:
#                 raise ValueError(msg_empty)
#             if len(variables) != len(set(variables)):
#                 raise ValueError(msg_dupes)
#         else:
#             if not isinstance(variables, (str, int)):
#                 raise ValueError(msg)
#     return variables


def find_eligible_categorical_variables(X: pl.DataFrame, n_unique: int = 20) -> List[Union[str, int]]:

    variables = [x for x in X.columns if X.schema[x] in 
                    [pl.datatypes.String,
                      pl.datatypes.Categorical,
                      pl.datatypes.Enum,
                      pl.datatypes.Utf8]]

    if len(variables) == 0:
        raise TypeError(
            "No categorical variables found in this dataframe. Please check "
            "variable format with pandas dtypes."
        )

    vars = []
    for v in variables:
        if X[v].n_unique() <= n_unique:
            vars.append(v)
    return vars


def is_variable_available(X, var):
    not_cols = []
    for i in var:
        if i not in X.columns:
            not_cols.append(i)

    not_cols = ",".join(["'"+x+"'" for x in not_cols])        

    if len(not_cols)>0:
        raise KeyError(f"Error: This variables {not_cols} is/are not available in your dataset")



"""Functions to check that the variables in a list are of a certain type."""


def check_numerical_variables(
    X: pd.DataFrame, variables: Variables
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are of type numerical.

    More details in the :ref:`User Guide <check_num_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset.

    variables : List
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the numerical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_numerical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = check_numerical_variables(X, variables=["var_num"])
    >>> var_
    ['var_num']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    if len(X[variables].select_dtypes(exclude="number").columns) > 0:
        raise TypeError(
            "Some of the variables are not numerical. Please cast them as "
            "numerical before using this transformer."
        )

    return variables


def check_categorical_variables(
    X: pd.DataFrame, variables: Variables
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are of type object or categorical.

    More details in the :ref:`User Guide <check_cat_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the categorical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_categorical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = check_categorical_variables(X, "var_cat")
    >>> var_
    ['var_cat']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    if len(X[variables].select_dtypes(exclude=["O", "category"]).columns) > 0:
        raise TypeError(
            "Some of the variables are not categorical. Please cast them as "
            "object or categorical before using this transformer."
        )

    return variables


def check_datetime_variables(
    X: pd.DataFrame,
    variables: Variables,
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are or can be parsed as datetime.

    More details in the :ref:`User Guide <check_datetime_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the datetime variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_datetime_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_date = check_datetime_variables(X, "var_date")
    >>> var_date
    ['var_date']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    # find non datetime variables, if any:
    non_datetime_vars = []
    for column in X[variables].select_dtypes(exclude="datetime"):
        if is_numeric(X[column]) or not _is_categorical_and_is_datetime(X[column]):
            non_datetime_vars.append(column)

    if len(non_datetime_vars) > 0:
        raise TypeError(
            "Some of the variables are not or cannot be parsed as datetime."
        )

    return variables


def check_all_variables(
    X: pd.DataFrame,
    variables: Variables,
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are in the dataframe.

    More details in the :ref:`User Guide <check_all_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_all_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> vars_all = check_all_variables(X, ['var_num', 'var_cat', 'var_date'])
    >>> vars_all
    ['var_num', 'var_cat', 'var_date']
    """
    if isinstance(variables, (str, int)):
        if variables not in X.columns.to_list():
            raise KeyError(f"The variable {variables} is not in the dataframe.")
        variables_ = [variables]

    else:
        if not set(variables).issubset(set(X.columns)):
            raise KeyError("Some of the variables are not in the dataframe.")

        variables_ = variables

    return variables_
  


