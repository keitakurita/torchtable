from .custom_types import *

def with_default(x: Optional[T], default: T) -> T:
    return default if x is None else x

def to_numpy_array(x: ArrayLike) -> np.array:
    """Extracts numpy array from an ArrayLike object"""
    if isinstance(x, pd.Series):
        return x.values
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)

def expand(a: OneorMore[T], n: int) -> Iterable[T]:
    """Turn 1 or n objects to n objects by repeating if necessary"""
    if isinstance(a, (tuple, list)):
        assert len(a) == n
        return a
    else:
        return [a for _ in range(n)]
