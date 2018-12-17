import pytest
import numpy as np
import pandas as pd

from torchtable import *

def test_pipe():
    op1 = LambdaOperator(lambda x: x + 1)
    op2 = op1 > LambdaOperator(lambda x: x ** 2)
    assert op2(1) == 4
    op3 = LambdaOperator(lambda x: x + 3)
    op2 > op3
    assert op3(2) == 12

def test_hook():
    op1 = LambdaOperator(lambda x: x + 3)
    op2 = LambdaOperator(lambda x: x * 2)
    op2 < op1
    assert op2(1) == 8
    op3 = LambdaOperator(lambda x: x ** 2)
    op3 < op2
    assert op3(1) == 64

def test_normalizer_gaussian():
    norm = Normalize("Gaussian")
    rng = np.random.RandomState(21)
    a = rng.normal(4, 10, (200, ))
    a_normed = norm(a)
    np.testing.assert_almost_equal(a_normed.mean(), 0.)
    np.testing.assert_almost_equal(a_normed.std(), 1.)

def test_normalizer_rank_gaussian():
    norm = Normalize("RankGaussian")
    rng = np.random.RandomState(21)
    a = rng.normal(4, 10, (200, ))
    a_normed = norm(a)
    np.testing.assert_almost_equal(a_normed.mean(), 0.)

def test_missing_filler():
    rng = np.random.RandomState(21)
    x = pd.Series(data=rng.normal(0, 1, (100, )))
    x[x < 0] = np.nan
    for mthd in ["median", "mean", "mode"]:
        filler = FillMissing(mthd)
        assert not pd.isnull(filler(x)).any()

def test_categorize():
    rng = np.random.RandomState(21)
    a = pd.Series(data=rng.randint(0, 20, (100, )))
    cat = Categorize()
    a_transformed = cat(a)

def test_categorize_min_max_freq():
    rng = np.random.RandomState(21)
    a = pd.Series(data=np.array([1, 2, 1, 4, 1, 2, 3, 3, 5]))
    cat = Categorize(min_freq=2, max_features=None, handle_unk=True)
    a_transformed = cat(a)
    assert (a_transformed[a == 4] == 0).all()
    assert (a_transformed[a == 5] == 0).all()
    assert (a_transformed[a == 1] != 0).all()

def test_categorize_unknown():
    rng = np.random.RandomState(21)
    a = pd.Series(data=np.array([0, 6, 7, 8, 9, 6, 3, 1, 2, 4]))
    cat = Categorize(min_freq=0, max_features=None, handle_unk=True)
    cat(pd.Series(data=np.arange(6)))
    a_transformed = cat(a, train=False)
    assert (a_transformed[a > 5] == 0).all()
    assert (a_transformed[a <= 5] > 0).all()