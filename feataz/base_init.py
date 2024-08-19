from typing import Any, List, Union, Dict
import polars as pl

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


def find_categorical_variables(X: pl.DataFrame) -> List[Union[str, int]]:

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

class BaseInit():
    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]], Dict] = None,
        ignore_format: bool = False,
    ) -> None:
        self.variables = validate_variables(variables)