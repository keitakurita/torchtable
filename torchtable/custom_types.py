import numpy as np
import pandas as pd
from typing import *

SeriesLike = Union[pd.Series, pd.DataFrame]
ArrayLike = Union[pd.Series, np.array]
T = TypeVar("T")
OneorMore = Union[T, Iterable[T]]
ColumnName = Union[str, List[str]]
