import numpy as np
import pandas as pd
import polars as pl
from typing import Any, List, Union, Dict
import polars as pl

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

def is_variable_available(data, var):
    not_cols = []
    for i in var:
        if i not in data.columns:
            not_cols.append(i)

    not_cols = ",".join(["'"+x+"'" for x in not_cols])        

    if len(not_cols)>0:
        raise KeyError(f"Error: This variables {not_cols} is/are not available in your dataset")
    


Variables = Union[None, int, str, List[Union[str, int]], Dict]


def validate_variables(variables: Variables) -> Any:
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

from typing import Any, List, Union, Dict
import polars as pl

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


def is_variable_available(data, var):
    not_cols = []
    for i in var:
        if i not in data.columns:
            not_cols.append(i)

    not_cols = ",".join(["'"+x+"'" for x in not_cols])        

    if len(not_cols)>0:
        raise KeyError(f"Error: This variables {not_cols} is/are not available in your dataset")