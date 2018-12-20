import logging
import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict
from scipy.special import erfinv
from scipy.stats import rankdata

from ..custom_types import *

from ..utils import *

logger = logging.getLogger(__name__)

# utils
def _most_frequent(x: np.ndarray):
    c = Counter(x)
    return c.most_common(1)[0][0]

class Operator:
    """
    Base class for all operators.
    Operators can be chained together by piping their outputs to new operators or hooking operators to other operators.
    Any number of operators can be chained to become a pipeline, which is itself just another operator.
    Subclasses should implement the `apply` method that defines the operation performed by the operator.
    
    Example:
        >>> class TimesThree(Operator):
        ...     def apply(self, x):
        ...         return x * 3
        >>> op = TimeThree()
        >>> op(4) # 4 * 3 = 12
        ... 12

        >>> class Square(Operator):
        ...     def apply(self, x):
                    return x ** 2
        >>> op = TimesThree() > Square()
        >>> op(2) # (2 * 3) ** 2 = 36
        ... 36
    """
    def __init__(self):
        self.before = None
        self.built = False

    def __gt__(self, op: 'Operator') -> 'Operator':
        """Syntactic sugar for piping"""
        return self.pipe(op)

    def __lt__(self, op: 'Operator') -> 'Operator':
        """Syntactic sugar for hooking"""
        return self.hook(op)
    
    def pipe(self, op: 'Operator') -> 'Operator':
        """Connect an operator after this operator. Returns the connected operator."""
        op.before = self
        return op
    
    def hook(self, op: 'Operator') -> 'Operator':
        """Connect an operator to the *beginning* of this pipeline. Returns self."""
        if self.before is not None:
            self.before.hook(op)
        else:
            self.before = op
        return self

    def apply(self, x: Any, train=True) -> Any:
        """
        Takes output of previous stage in the pipeline and produces output. Override in subclasses.
        Kwargs:
            train: If true, this operator will "train" on the input. 
            In other words, the internal parameters of this operator may change to fit the given input.
        """
        return x
    
    def __call__(self, x, **kwargs):
        if self.before is not None:
            return self.apply(self.before(x, **kwargs), **kwargs)
        else:
            return self.apply(x, **kwargs)

class LambdaOperator(Operator):
    """
    Generic operator for stateless operation.
    Args:
        func: Function to apply to input.
    """
    def __init__(self, func: Callable[[T], T]):
        super().__init__()
        self.func = func
    
    def apply(self, x: T, train=True) -> Any:
        return self.func(x)

class TransformerOperator(Operator):
    """
    Wrapper for any stateful transformer with fit and transform methods.
    Args:
        transformer: Any object with a `fit` and `transform` method.
    Example:
        >>> op = TransformerOperator(sklearn.preprocessing.StandardScaler())
    """
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
    
    def build(self, x: Any) -> None:
        self.transformer.fit(x)
    
    def apply(self, x: Any, train=True):
        if train: self.build(x)
        return self.transformer.transform(x)

class _Normalizer:
    _methods = set(["Gaussian", "RankGaussian"])

    def __init__(self, method):
        self.method = method
        if method is not None and method not in self._methods:
            raise ValueError(f"Invalid normalization method {method}")
    
    def fit(self, x: pd.Series):
        if self.method == "Gaussian":
            self.mean, self.std = x.mean(), x.std()
        elif self.method == "RankGaussian":
            # TODO: store state
            pass
        return self

    def transform(self, x: pd.Series) -> pd.Series:
        if self.method == "Gaussian":
            return (x - self.mean) / (self.std + 1e-8)
        elif self.method == "RankGaussian":
            # TODO: store state
            # prevent divergence to infinity by restricting normalized ranks to range[-0.99, 0.99]
            x = (rankdata(x) / len(x) - 0.5) * 0.99 * 2
            x = erfinv(x)
            return (x - x.mean())
        else:
            return x

class Normalize(TransformerOperator):
    """
    Normalizes a numeric field.
    Args:
        method: Method of normalization (choose from the following):
        - None: No normalization will be applied (same as noop)
        - 'Gaussian': Subtracts mean and divides by the standard deviation
        - 'RankGaussian': Assigns elements to a Gaussian distribution based on their rank.
    """
    def __init__(self, method: Optional[str]):
        super().__init__(_Normalizer(method))

class _MissingFiller:
    _method_mapping = {
        "median": lambda x: x.median(),
        "mean": lambda x: x.mean(),
        "mode": lambda x: _most_frequent(x.dropna()),
    }
    
    def __init__(self, method):
        if callable(method):
            self.method = method
        elif method in self._method_mapping:
            self.method = self._method_mapping[method]
        elif method is None:
            self.method = None
        else:
            raise ValueError(f"Invalid method of filling missing data: {method}")
        self.na_mapping = {}

    def fit(self, x: pd.Series) -> '_MissingFiller':
        if self.method is not None:
            self.fill_value = self.method(x)
        return self
    
    def transform(self, x: pd.Series) -> pd.Series:
        if self.method is not None:
            return x.fillna(self.fill_value)
        else:
            return x
        
class FillMissing(TransformerOperator):
    """
    Fills missing values according to `method`
    Args:
        method: Method of filling missing values. Options:
        - None: Do not fill missing values
        - 'median': Fill with median
        - 'mean': Fill with mean
        - 'mode': Fill with mode. Effective for categorical fields.
        - (any callable): The output of the callable will be used to fill the missing values
    """
    def __init__(self, method: Union[Callable, str]):
        super().__init__(_MissingFiller(method))

class Vocab:
    """Mapping from category to integer id"""
    def __init__(self, min_freq=0, max_features=None,
                 handle_unk: Optional[bool]=False, nan_as_unk=False):
        self.min_freq = min_freq
        self.max_features = max_features
        self.handle_unk = with_default(handle_unk, min_freq > 0 or max_features is not None)
        self.nan_as_unk = nan_as_unk

        if not self.handle_unk and (max_features is not None or min_freq > 0):
            logger.warn("""Setting max_features or min_freq will potentially cause some categories to become unknown.
            Set handle_unk to True to handle categories left out due to max_features or min_freq being set.
            """)
        
        if not handle_unk and nan_as_unk:
            raise ValueError("""Setting nan_as_unk=True requires the vocabulary to be able to handle unk.
            Set handle_unk=True if setting nan_as_unk to True.""")
    
    def fit(self, x: pd.Series) -> 'Vocab':
        """Construct the mapping"""
        counter = Counter()
        for v in x:
            if self.nan_as_unk and np.isnan(x): continue
            counter[v] += 1
        
        self.index = defaultdict(int)
        # if handle unknown category, reserve 0 for unseen categories
        idx = 1 if self.handle_unk else 0
        for k, c in counter.most_common(self.max_features):
            if c < self.min_freq: break
            self.index[k] = idx; idx += 1
        return self
    
    def _get_index(self, x):
        if x not in self.index and not self.handle_unk:
            raise ValueError("Found category not in vocabulary. Try setting handle_unk to True.")
        else:
            return self.index[x]
    
    def transform(self, x: pd.Series) -> pd.Series:
        return x.apply(self._get_index)

    def __len__(self):
        return len(self.index) + (1 if self.handle_unk else 0)
    
class Categorize(TransformerOperator):
    """
    Converts categorical data into integer ids
    Args:
        min_freq: Minimum frequency required for a category to receive a unique id.
        Any categories with a lower frequency will be treated as unknown categories.
        max_features: Maximum number of unique categories to store. If larger than the number of actual categories,
        the categories with the highest frequencies will be chosen. If None, there will be no limit on the number of categories.
        handle_unk: Whether to allocate a unique id to unknown categories. 
        If you expect to see categories that you did not encounter in your training data, you should set this to True.
        If None, handle_unk will be set to True if min_freq > 0 or max_features is not None, otherwise it will be False.
    """
    def __init__(self, min_freq: int=0, max_features: Optional[int]=None,
                 handle_unk: Optional[bool]=None):
        super().__init__(Vocab(min_freq=min_freq, max_features=max_features,
                               handle_unk=handle_unk))
    
    @property
    def vocab_size(self):
        return len(self.transformer)

class ToTensor(Operator):
    """
    Convert input to a `torch.tensor`
    Args:
        dtype: The dtype of the output tensor
    """
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
    
    def apply(self, x: ArrayLike, device: Optional[torch.device]=None, train=True) -> torch.tensor:
        arr = to_numpy_array(x)
        # convert dtype to PyTorch compatible type
        if arr.dtype == np.bool_:
            arr = arr.astype("int")
        return torch.tensor(arr, dtype=self.dtype, device=device)