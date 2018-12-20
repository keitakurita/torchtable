import pytest

from torchtable import *
from torchtable.operator import *
from torchtable.field import *

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

def test_batch_transform():
    """Smoke test for batch transformations"""
    rng = np.random.RandomState(21)
    a = pd.Series(data=rng.normal(0, 1, (100, )))
    fld = NumericField()
    tsr = fld.transform_batch(fld.transform(a))

def test_field_metadata():
    fld = NumericField(metadata={"foo": "bar"})
    assert fld.metadata["foo"] == "bar"

def test_field_collection():
    fld0 = NumericField()
    fld1 = CategoricalField()
    flds = FieldCollection(fld0, fld1)
    assert len(flds) == 2
    assert flds[0] == fld0
    assert flds[1] == fld1

def test_namespace():
    fld0 = NumericField()
    fld1 = CategoricalField(name="bar")
    flds = FieldCollection(fld0, fld1, namespace="foo")
    assert fld0.name == "foo/_0"
    assert fld1.name == "foo/bar"
    flds.name = "hoge"
    assert fld0.name == "hoge/_0"
    assert fld1.name == "hoge/bar"
    
    fld0 = NumericField()
    fld1 = CategoricalField(name="bar")
    flds = FieldCollection(fld0, fld1)
    flds.name = "hoge"
    flds.name = "hoge"
    assert fld0.name == "hoge/_0"
    assert fld1.name == "hoge/bar"

def test_index():
    fld = NumericField()
    np.testing.assert_almost_equal(fld.index(np.arange(10), [0, 3, 5]), np.array([0, 3, 5]))

def test_index_fieldcollection():
    flds = FieldCollection(NumericField(), NumericField())
    arr1, arr2 = flds.index([np.array([5, 4, 2, 3, 1]), np.array([1, 2, 3, 4, 5])], [1, 4, 2])
    np.testing.assert_almost_equal(arr1, np.array([4, 1, 2]))
    np.testing.assert_almost_equal(arr2, np.array([2, 5, 3]))

def test_fieldcollection_transform():
    flds = FieldCollection(Field(LambdaOperator(lambda x: x * 2)), Field(LambdaOperator(lambda x: x + 3)))
    assert flds.transform(1) == [2, 4]

def test_unknown_cat():
    fld = CategoricalField(name="hoge", handle_unk=False)
    a = pd.Series(data=np.array([1, 2, 3]))
    fld.transform(a)
    b = pd.Series(data=np.array([3, 4]))
    with pytest.raises(UnknownCategoryError):
        fld.transform(b, train=False)