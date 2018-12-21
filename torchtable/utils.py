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

def expand(a: OneorMore, n: int) -> Iterable:
    """Turn 1 or n objects to n objects by repeating if necessary"""
    if isinstance(a, (tuple, list)):
        assert len(a) == n
        return a
    else:
        return [a for _ in range(n)]

def flat_filter(itr: Iterable[Union[T, Iterable[T]]], predicate: Callable[[T], bool]) -> Iterable[T]:
    for x in itr:
        if isinstance(x, (tuple, list)):
            for x_ in x:
                if predicate(x_): yield x_
        else:
            if predicate(x): yield x

def apply_oneormore(func, x: OneorMore[T]) -> OneorMore[Any]:
    """
    Returns multiple outputs for multiple inputs and a single output for a single input,
    applying the same function in either case.
    """
    if isinstance(x, (tuple, list)):
        return [func(item, idx) for idx, item in enumerate(x)]
    else:
        return func(x, -1)

def fold_oneormore(func: Callable[[T, T], T], x: OneorMore[T], init: T) -> Any:
    """
    Aggregates multiple inputs by folding, simply applies for a single input.
    """
    if isinstance(x, (tuple, list)):
        r = init
        for v in x: r = func(r, x)
        return r
    else:
        return func(init, x)
