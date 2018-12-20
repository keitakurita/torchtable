import logging
import numpy as np
import pandas as pd

import torch.utils.data
from pathlib import Path
import warnings

from ..custom_types import *

from ..utils import *
from ..operator import Operator, LambdaOperator, FillMissing, Categorize, Normalize, ToTensor, UnknownCategoryError

logger = logging.getLogger(__name__)

class Field:
    """
    A single field in the output mini batch. A Field acts as a continaer for all relevant information regarding an output in the output mini batch.
    Primarily, it stores a pipeline to apply to a column/set of columns in the input.
    It also stores a pipeline for converting the input batch to an appropriate type for the downstream model (generally a torch.tensor).
    This class can directly be instantiated with a custom pipeline but is generally used as a subclass for other fields.
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
        metadata: Additional data about the field to store. 
        Use cases include adding data about model parameters (e.g. size of embeddings for this field).
    """
    def __init__(self, pipeline: Operator, name: Optional[str]=None,
                 is_target: bool=False, continuous: bool=True,
                 categorical: bool=False, batch_pipeline: Optional[Operator]=None,
                 dtype: Optional[torch.dtype]=None, metadata: dict={}):
        self.pipeline = pipeline
        self.name = name
        self.is_target = is_target
        if categorical and continuous:
            raise ValueError("""A field cannot be both continuous and categorical. 
            If you want both a categorical and continuous representation, consider using multiple fields.""")
        self.continuous, self.categorical = continuous, categorical

        if dtype is not None and batch_pipeline is not None:
            logger.warning("""Setting a custom batch pipeline will cause this field to ignore the dtype argument.
            If you want to manually set the dtype, consider attaching a ToTensor operation to the pipeline.""")
        dtype = with_default(dtype, torch.long if self.categorical else torch.float)
        self.batch_pipeline = with_default(batch_pipeline, ToTensor(dtype))
        self.metadata = metadata
        
    def transform(self, x: pd.Series, train=True) -> ArrayLike:
        """
        Method to process the input column during construction of the dataset.
        Kwargs:
            train: If true, this transformation may change some internal parameters of the pipeline.
            For instance, if there is a normalization step in the pipeline, the mean and std will be computed on the current input.
            Otherwise, the pipeline will use statistics computed in the past.
        """
        return self.pipeline(x, train=train)

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.name}]"
    
    def transform_batch(self, x: ArrayLike, device: Optional[torch.device]=None, 
                        train: bool=True) -> torch.tensor:
        """Method to process batch input during loading of the dataset."""
        return self.batch_pipeline(x, device=device, train=train)

    def index(self, example: ArrayLike, idx) -> ArrayLike:
        """
        Wrapper for indexing. The field must provide the ability to index via a list for batching later on.
        """
        return example[idx]

class IdentityField(Field):
    """
    A field that does not modify the input.
    """
    def __init__(self, name=None, is_target=False, continuous=True, categorical=False, metadata={}):
        super().__init__(LambdaOperator(lambda x: x), name=name,
                         is_target=is_target, continuous=continuous, categorical=categorical, metadata=metadata)

class NumericField(Field):
    """
    A field corresponding to a continous, numerical output (e.g. price, distance, etc.)
    Args:
        fill_missing: The method of filling missing values. See the `FillMissing` operator for details.
        normalization: The method of normalization. See the `Normalize` operator for details.
    """
    def __init__(self, name=None,
                 fill_missing="median", normalization="Gaussian",
                 is_target=False, metadata={}):
        pipeline = FillMissing(fill_missing) > Normalize(normalization)
        super().__init__(pipeline, name, is_target, continuous=True, categorical=False, metadata=metadata)

class CategoricalField(Field):
    """
    A field corresponding to a categorica, discrete output (e.g. id, group, gender)
    Args:
        See the `Categorize` operator for more details.
    """
    def __init__(self, name=None, min_freq=0, max_features=None,
                 handle_unk=None, is_target=False, metadata: dict={}):
        pipeline = Categorize(min_freq=min_freq, max_features=max_features,
                              handle_unk=handle_unk)
        self.vocab = pipeline.transformer
        super().__init__(pipeline, name, is_target, continuous=False, categorical=True, metadata=metadata)
    
    def transform(self, x: pd.Series, train=True) -> ArrayLike:
        try:
            return super().transform(x, train=train)
        except UnknownCategoryError:
            raise UnknownCategoryError(f"Unknown category encountered in {self.name}. Consider setting handle_unk=True.")
    
    @property
    def cardinality(self):
        """The number of unique outputs."""
        return len(self.vocab)

class FieldCollection(list):
    """
    """
    def __init__(self, *args, flatten: bool=False, namespace: Optional[str]=None):
        for a in args: self.append(a)
        self.flatten = flatten
        self.namespace = None
        self.set_namespace(namespace)
    
    def index(self, examples: List[ArrayLike], idx) -> List[ArrayLike]:
        return [fld.index(ex, idx) for fld, ex in zip(self, examples)]

    @property
    def name(self) -> str:
        return self.namespace

    def set_namespace(self, nm: str) -> None:
        """Set names of inner fields as well"""
        old_namespace = self.namespace
        self.namespace = nm
        for i, fld in enumerate(self):
            if fld.name is None: 
                fld.name = f"{self.namespace}/_{i}"
            else:
                if old_namespace is not None and fld.name.startswith(f"{old_namespace}/"):
                    fld.name = fld.name[len(old_namespace)+1:]
                fld.name = f"{self.namespace}/{fld.name}"    
    @name.setter
    def name(self, nm: str):
        self.set_namespace(nm)
    
    def transform(self, *args, **kwargs) -> list:
        """Applies transform with each field and returns a list"""
        return [fld.transform(*args, **kwargs) for fld in self]