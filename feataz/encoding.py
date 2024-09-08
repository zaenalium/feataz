import polars as pl
from datetime import datetime
import pandas as pd
from typing import Any, List, Union, Dict

from feataz.utils import check_data, is_variable_available, find_eligible_categorical_variables

from typing import List, Optional, Dict, Union

from pydantic import BaseModel


df = pl.read_csv("data/train.csv")

class OneHot():
    """One Hot Encoder class."""

    def __init__(self, 
                 variable:  Union[None, int, str, List[Union[str, int]], Dict] = None, 
                 n_top_category: Optional[int] = None, 
                 drop_first: bool = False, 
                 keep_original: bool = False
                 ) -> None:
        """Init.

        Args:
            variable (str | list | dict): list of features to encode
            n_top_category (int): drop or keep the one hot encoded column
            drop_first
            keep_original
        """
        self.n_top_category = n_top_category
        self.drop_first = drop_first
        self.variable = variable
        self.keep_original = keep_original

    def fit(self, data: Union[pl.DataFrame, pd.DataFrame]):
        """
        """
        
        df = check_data(data)

        most_freq = {}
        if self.variable:
            if type(self.variable) == dict:
                var = list(self.variable.keys)
                for j in var:
                    n_top = self.variable[j]
                    most_freq[j] = df[j].value_counts().sort(by = 'count',
                                         descending = True)[:n_top][j].to_list()
            else:
                var = list(self.variable)

        else:
            var = find_eligible_categorical_variables(df)

        if self.n_top_category:
            for j in var:
                most_freq[j] = df[j].value_counts(
                                    ).sort(by = 'count',
                                     descending = True)[:self.n_top_category][j].to_list()
        
        is_variable_available(df, var)

        self.var = var
        self.most_freq = most_freq
        return self

    def transform(self, data):
        df = check_data(data)

        is_variable_available(df, self.var)

        if not self.most_freq:
            dum = df[self.var].to_dummies(drop_first = self.drop_first)
        else:
            dum = pl.DataFrame([])
            for i in self.var:
                items = [str(x) for x in self.most_freq[i]]
                tmp = df.select(i).with_columns(pl.when(pl.col(i).cast(str).is_in(items) == False)
                                     .then(pl.lit(None))
                                     .otherwise(pl.col(i))
                                     .alias(i)
                                     )
                dum = pl.concat([dum, tmp], how = 'horizontal')
        dum = dum.to_dummies(drop_first = self.drop_first)
        if not self.keep_original:
            result = pl.concat([df.drop(self.var),dum ], how = 'horizontal')
        else:
            result = pl.concat([df,dum ], how = 'horizontal')
            
        return result


### Ordinal Encoding

class Ordinal():
    """One Hot Encoder class."""

    def __init__(self, 
                 variable:  Union[None, int, str, List[Union[str, int]], Dict] = None, 
                 n_top_category: Optional[int] = None, 
                 drop_first: bool = False, 
                 keep_original: bool = False
                 ) -> None:
        """Init.

        Args:
            variable (str | list | dict): list of features to encode
            n_top_category (int): drop or keep the one hot encoded column
            drop_first
            keep_original
        """
        self.n_top_category = n_top_category
        self.drop_first = drop_first
        self.variable = variable
        self.keep_original = keep_original
