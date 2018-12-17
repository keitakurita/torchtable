import pytest

from torchtable import *

def test_field():
    fld = Field(LambdaOperator(lambda x: x + 1) > LambdaOperator(lambda x: x ** 2))
    assert fld.transform(1) == 4

def test_numeric_field():
    rng = np.random.RandomState(21)
    x = pd.Series(data=rng.normal(0, 1, (100, )))
    x[x < 0] = np.nan
    for mthd in ["median", "mean", "mode"]:
        fld = NumericField(fill_missing=mthd)
        assert not pd.isnull(fld.transform(x)).any()
    
    fld = NumericField(fill_missing=None)
    assert pd.isnull(fld.transform(x)).any()

def test_numeric_field_norm():
    rng = np.random.RandomState(21)
    x = pd.Series(data=rng.normal(-1, 4, (100, )))
    fld = NumericField(fill_missing=None, normalization="Gaussian")
    np.testing.assert_almost_equal(fld.transform(x).mean(), 0.)
    np.testing.assert_almost_equal(fld.transform(x).std(), 1.)
    
    fld = NumericField(fill_missing=None, normalization="RankGaussian")
    np.testing.assert_almost_equal(fld.transform(x).mean(), 0.)

def test_numeric_joint():
    """Smoke test for NumericField with various settings"""
    rng = np.random.RandomState(21)
    x = pd.Series(data=rng.normal(2, 0.5, (100, )))
    for fill_mthd in ["median", "mean", "mode"]:
        for norm_mthd in [None, "Gaussian", "RankGaussian"]:
            fld = NumericField(fill_missing=fill_mthd, normalization=norm_mthd)
            fld.transform(x)

def test_categorical_field():
    """Smoke test for categorical field with default settings"""
    rng = np.random.RandomState(21)
    x = pd.Series(data=rng.randint(-3, 15, (100, )))
    fld = CategoricalField(handle_unk=False)
    assert fld.transform(x).nunique() == len(fld.vocab)
    assert fld.transform(x).nunique() == fld.cardinality

def test_datetime_fields():
    """Smoke test for fields"""
    x = pd.to_datetime(pd.DataFrame({'year': [2015, 2016, 2015, 2017, 2020], 'month': [2, 3, 4, 5, 1], 
                                 'day': [4, 5, 10, 29, 30], 'hour': [2, 3, 12, 11, 5]}))
    for fld_type in [DayofWeekField, DayField, MonthStartField, MonthEndField, HourField]:
        assert not pd.isnull(fld_type().transform(x)).any()

def test_date_fields():
    x = pd.to_datetime(pd.DataFrame({'year': [2011, 1995, 2015, 2017, 2030], 'month': [12, 9, 7, 5, 10], 
                                     'day': [14, 13, 9, 19, 1]}))
    for fld_type in [DayofWeekField, DayField, MonthStartField, MonthEndField]:
        assert not pd.isnull(fld_type().transform(x)).any()

def test_to_tensor():
    """Smoke test for the to_tensor method"""
    rng = np.random.RandomState(21)
    x = pd.Series(data=rng.normal(0, 1, (100, )))
    fld = NumericField()
    tsr = fld.to_tensor(fld.transform(x), device=None)

def test_to_tensor_bool():
    """Smoke test for the to_tensor method for boolean data"""
    x = pd.Series(data=np.array([True, False, True, False]))
    fld = NumericField()
    tsr = fld.to_tensor(fld.transform(x), device=None)