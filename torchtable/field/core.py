import numpy as np
import pandas as pd

import torch.utils.data
from pathlib import Path
import warnings

from ..operator import Operator, LambdaOperator, FillMissing, Categorize, Normalize

from ..custom_types import *

from ..utils import *

class Field:
    """A single field in the output mini batch that represents a set of operations to perform to a set of inputs.
    Base class for other fields. Can be instantiated by passing a pipeline."""
    def __init__(self, pipeline: Operator, name: Optional[str]=None,
                 is_target: bool=False, continuous: bool=True,
                 categorical: bool=False, dtype: Optional[torch.dtype]=None):
        self.pipeline = pipeline
        self.name = name
        self.is_target = is_target
        if categorical and continuous:
            raise ValueError("""A field cannot be both continuous and categorical. 
            If you want both a categorical and continuous representation, consider using multiple fields.""")
        self.continuous = continuous
        self.categorical = categorical
        self.dtype = with_default(dtype, torch.long if self.categorical else torch.float)
    
    def transform(self, x: pd.Series, train=True) -> ArrayLike:
        """Method to process input column during construction of the dataset."""
        return self.pipeline(x, train=train)

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name}]"
    
    def to_tensor(self, x: ArrayLike, device: torch.device, train=True) -> torch.tensor:
        """Method to convert input batch to a `torch.tensor` during batching of input."""
        arr = to_numpy_array(x)
        # convert dtype to PyTorch compatible type
        if arr.dtype == np.bool_:
            arr = arr.astype("int")
        return torch.tensor(arr, dtype=self.dtype, device=device)

class IdentityField(Field):
    def __init__(self, name=None, is_target=False, continuous=True, categorical=False):
        super().__init__(LambdaOperator(lambda x: x), name=name,
                         is_target=is_target, continuous=continuous, categorical=categorical)

class NumericField(Field):
    def __init__(self, name=None,
                 fill_missing="median", normalization="Gaussian",
                 is_target=False):
        pipeline = FillMissing(fill_missing) > Normalize(normalization)
        super().__init__(pipeline, name, is_target, continuous=True, categorical=False)

class CategoricalField(Field):
    def __init__(self, name=None, min_freq=0, max_features=None,
                 handle_unk=None, is_target=False):
        pipeline = Categorize(min_freq=min_freq, max_features=max_features,
                              handle_unk=handle_unk)
        self.vocab = pipeline.transformer
        super().__init__(pipeline, name, is_target, continuous=False, categorical=True)
    
    @property
    def cardinality(self):
        return len(self.vocab)

class DatetimeFeatureField(Field):
    def __init__(self, func: Callable[[pd.Series], pd.Series], fill_missing: Optional[str]=None,
                 name=None, is_target=False, continuous=False):
        pipeline = (LambdaOperator(lambda s: pd.to_datetime(s))
                    > FillMissing(method=fill_missing) 
                    > LambdaOperator(lambda s: func(s.dt)))
        super().__init__(pipeline, name=name, is_target=is_target, continuous=continuous, categorical=not continuous)

class DayofWeekField(DatetimeFeatureField):
    def __init__(self, **kwargs):
        super().__init__(lambda x: x.dayofweek, **kwargs)

class DayField(DatetimeFeatureField):
    def __init__(self, **kwargs):
        super().__init__(lambda x: x.day, **kwargs)

class MonthStartField(DatetimeFeatureField):
    def __init__(self, **kwargs):
        super().__init__(lambda x: x.is_month_start, continuous=False, **kwargs)

class MonthEndField(DatetimeFeatureField):
    def __init__(self, **kwargs):
        super().__init__(lambda x: x.is_month_end, **kwargs)

class HourField(DatetimeFeatureField):
    def __init__(self, **kwargs):
        super().__init__(lambda x: x.hour, **kwargs)

def date_fields(**kwargs) -> List[DatetimeFeatureField]:
    return [DayofWeekField(**kwargs), DayField(**kwargs),
            MonthStartField(**kwargs), MonthEndField(**kwargs),
           ]

def datetime_fields(**kwargs) -> List[DatetimeFeatureField]:
    return [DayofWeekField(**kwargs), DayField(**kwargs),
            MonthStartField(**kwargs), MonthEndField(**kwargs),
            HourField(**kwargs),
           ]