import logging
import numpy as np
import pandas as pd

import torch.utils.data
from pathlib import Path
import warnings

from ..custom_types import *

from ..utils import *
from ..operator import Operator, LambdaOperator, FillMissing, Categorize, Normalize, ToTensor

logger = logging.getLogger(__name__)

class Field:
    """
    A single field in the output mini batch. 
    A Field object wraps a pipeline to apply to a column/set of columns in the input.
    This class can directly be instantiated with a custom pipeline.
    Example:
        >>> fld = Field(LambdaOperator(lambda x: x + 1) > LambdaOperator(lambda x: x ** 2))
        >>> fld.transform(1)
        ... 9
    Args:
        pipeline: An operator representing the set of operations mapping the input column to the output.
        This transformation will be applied during the construction of the dataset. 
        If the pipeline is resource intensive and applying it all at once is unrealistic, consider deferring some of the processing to `batch_pipeline`.
    Kwargs:
        is_target: Whether the field is an input or target field. Affects default batching behavior.
        continuous: Whether the output is continuous.
        categorical: Whether the output is categorical/discrete.
        batch_pipeline: The transformation to apply to this field during batching.
        By default, this will simply be an operation to transform the input to a tensor to feed to the model.
        This can be set to any Operator that the user wishes so that arbitrary transformations (e.g. padding, noising) can be applied during data loading.
        dtype: The output tensor dtype. Only relevant when batch_pipeline is None (using the default pipeline).
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
        """Method to process the input column during construction of the dataset."""
        return self.pipeline(x, train=train)

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name}]"
    
    def batch_transform(self, x: ArrayLike, device: Optional[torch.device]=None, 
                        train: bool=True) -> torch.tensor:
        """Method to process batch input during loading of the dataset."""
        return self.batch_pipeline(x, device=device, train=train)

class IdentityField(Field):
    """
    A field that does not modify the input.
    """
    def __init__(self, name=None, is_target=False, continuous=True, categorical=False):
        super().__init__(LambdaOperator(lambda x: x), name=name,
                         is_target=is_target, continuous=continuous, categorical=categorical)

class NumericField(Field):
    """
    A field corresponding to a continous, numerical output (e.g. price, distance, etc.)
    Args:
        fill_missing: The method of filling missing values. See the `FillMissing` operator for details.
        normalization: The method of normalization. See the `Normalize` operator for details.
    """
    def __init__(self, name=None,
                 fill_missing="median", normalization="Gaussian",
                 is_target=False):
        pipeline = FillMissing(fill_missing) > Normalize(normalization)
        super().__init__(pipeline, name, is_target, continuous=True, categorical=False)

class CategoricalField(Field):
    """
    A field corresponding to a categorica, discrete output (e.g. id, group, gender)
    Args:
        See the `Categorize` operator for more details.
    """
    def __init__(self, name=None, min_freq=0, max_features=None,
                 handle_unk=None, is_target=False):
        pipeline = Categorize(min_freq=min_freq, max_features=max_features,
                              handle_unk=handle_unk)
        self.vocab = pipeline.transformer
        super().__init__(pipeline, name, is_target, continuous=False, categorical=True)
    
    @property
    def cardinality(self):
        """The number of unique outputs."""
        return len(self.vocab)

class DatetimeFeatureField(Field):
    """
    A generic field for constructing features from datetime columns.
    Args:
        func: Feature construction function
    """
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
    """The default set of fields for feature engineering using a field with date information"""
    return [DayofWeekField(**kwargs), DayField(**kwargs),
            MonthStartField(**kwargs), MonthEndField(**kwargs),
           ]

def datetime_fields(**kwargs) -> List[DatetimeFeatureField]:
    """The default set of fields for feature engineering using a field with date and time information"""
    return [DayofWeekField(**kwargs), DayField(**kwargs),
            MonthStartField(**kwargs), MonthEndField(**kwargs),
            HourField(**kwargs),
           ]