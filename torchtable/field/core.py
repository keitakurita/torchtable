import logging
import numpy as np
import pandas as pd

import torch.utils.data
from pathlib import Path
import warnings

from ..operator import Operator, LambdaOperator, FillMissing, Categorize, Normalize, ToTensor

from ..custom_types import *

from ..utils import *

logger = logging.getLogger(__name__)

class Field:
    """
    A single field in the output mini batch. 
    A Field object wraps a pipeline to apply to a column/set of columns in the input.
    This class can directly be instantiated with a custom pipeline.
    Example:
        >>> fld = Field(LambdaOperator(lambda x: x + 1) > LambdaOperator(lambda x: x ** 2))
    Args:
        pipeline: An operator representing the set of operations mapping the input column to the output.
    Kwargs:
        
    """
    def __init__(self, pipeline: Operator, name: Optional[str]=None,
                 is_target: bool=False, continuous: bool=True,
                 categorical: bool=False, batch_pipeline: Optional[Operator]=None,
                 dtype: Optional[torch.dtype]=None):
        self.pipeline = pipeline
        self.name = name
        self.is_target = is_target
        if categorical and continuous:
            raise ValueError("""A field cannot be both continuous and categorical. 
            If you want both a categorical and continuous representation, consider using multiple fields.""")
        self.continuous = continuous
        self.categorical = categorical
        if dtype is not None and batch_pipeline is not None:
            logger.warning("""Setting a custom batch pipeline will cause this field to ignore the dtype argument.
            If you want to manually set the dtype, consider attaching a ToTensor operation to the pipeline.""")
        dtype = with_default(dtype, torch.long if self.categorical else torch.float)
        self.batch_pipeline = with_default(batch_pipeline, ToTensor(dtype))
        
    def transform(self, x: pd.Series, train=True) -> ArrayLike:
        """Method to process input column during construction of the dataset."""
        return self.pipeline(x, train=train)

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name}]"
    
    def batch_transform(self, x: ArrayLike, device: Optional[torch.device]=None, 
                        train: bool=True) -> torch.tensor:
        """Method to process batch input during loading of the dataset."""
        return self.batch_pipeline(x, device=device, train=train)

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