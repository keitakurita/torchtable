import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from collections import Counter, defaultdict
from scipy.special import erfinv
from scipy.stats import rankdata

from typing import *

from ..custom_types import *

from ..utils import *

logger = logging.getLogger(__name__)

# utils
def _most_frequent(x: np.ndarray):
    c = Counter(x)
    return c.most_common(1)[0][0]

class Operator:
    """Base class for all operators. 
    Operators can be chained together by piping their outputs to new operators or hooking operators to other operators.
    Any number of operators can be chained to become a pipeline, which is itself just another operator.
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
        """Connect an operator after this operator. Returns the connected operator.
        """
        op.before = self
        return op
    
    def hook(self, op: 'Operator') -> 'Operator':
        """Connect an operator to the beginning of this pipeline. Returns self."""
        if self.before is not None:
            self.before.hook(op)
        else:
            self.before = op
        return self

    def apply(self, x: Any, train=True) -> Any:
        """Takes output of previous stage in the pipeline and produces output.
        Override in subclasses."""
        return None
    
    def __call__(self, x, **kwargs):
        if self.before is not None:
            return self.apply(self.before(x, **kwargs), **kwargs)
        else:
            return self.apply(x, **kwargs)

class LambdaOperator(Operator):
    """Generic operator for stateless operation."""
    def __init__(self, func: Callable[[T], T]=None):
        super().__init__()
        self.func = func
    
    def apply(self, x: Any, train=True) -> Any:
        return self.func(x)

class TransformerOperator(Operator):
    """Wrapper for any stateful transformer with fit and transform methods"""
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
    """Normalizes a numeric field. Available normalization methods:
    - Gaussian: Subtracts mean and divides by the standard deviation
    - RankGaussian: Assigns elements to a Gaussian distribution based on their rank.
    """
    def __init__(self, method):
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
    def __init__(self, method):
        super().__init__(_MissingFiller(method))

class Vocab:
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
        return len(self.index)
    
class Categorize(TransformerOperator):
    """Converts categorical data into integer ids"""
    def __init__(self, min_freq=0, max_features=None,
                 handle_unk=False):
        super().__init__(Vocab(min_freq=min_freq, max_features=max_features,
                               handle_unk=handle_unk))
    
    @property
    def vocab_size(self):
        return len(self.transformer)