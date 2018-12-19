import pytest

from torchtable import *
from torchtable.field import *
from torchtable.operator import *
from torchtable.dataset import *

def test_basic():
    df = pd.DataFrame({"a": [50, 40, 30, 20, 10],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": CategoricalField(),
        "b": NumericField(fill_missing=None, normalization=None),
    })
    assert len(ds) == len(df)
    for i in range(len(ds)):
        example = ds[i]
        assert "a" in example
        assert "b" in example
        assert example["a"] != df.iloc[i]["a"]
        assert example["b"] == df.iloc[i]["b"]

def test_multiple_fields():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5], 
                       "c": [1, 1, 1, 1, 1]})
    ds = TabularDataset.from_df(df, fields={
        "a": CategoricalField(max_features=100),
        "b": [NumericField(normalization="Gaussian"), Field(LambdaOperator(lambda x: x * 2))],
        "c": None,
    })
    assert len(ds) == len(df)
    assert len(ds.fields) == 2
    for i in range(len(ds)):
        example = ds[i]
        assert "a" in example
        assert "b" in example
        assert len(example) == 2

def test_index_with_list():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": CategoricalField(max_features=100),
        "b": [NumericField(normalization="Gaussian"), IdentityField()],
    })
    assert len(ds.fields) == 2
    list_idx = [0, 1, 3, 4]
    examples = ds[list_idx]
    assert len(examples) == 2
    assert len(examples["a"]) == 4
    assert len(examples["b"]) == 2
    assert len(examples["b"][0]) == 4
    assert len(examples["b"][1]) == 4
    assert (examples["b"][1].values == df.iloc[list_idx]["b"].values).all()

def test_from_dfs():
    df1 = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [-1., -2, -3.]})
    df3 = pd.DataFrame({"a": [3, 2], "b": [-1., -2]})
    # all present
    train, val, test = TabularDataset.from_dfs(df1, val_df=df2, test_df=df3, fields={
        "a": CategoricalField(),
        "b": [NumericField(normalization="Gaussian"), CategoricalField(handle_unk=True)],
    })
    # only train and val
    train, val = TabularDataset.from_dfs(df1, val_df=df2, test_df=None, fields={
        "a": CategoricalField(),
        "b": [NumericField(normalization="Gaussian"), CategoricalField(handle_unk=True)],
    })
    # only train and test
    train, test = TabularDataset.from_dfs(df1, val_df=None, test_df=df3, fields={
        "a": CategoricalField(),
        "b": [NumericField(normalization="Gaussian"), CategoricalField(handle_unk=True)],
    })

def test_from_dfs_with_target():
    df1 = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [-1., -2, -3.]})
    df3 = pd.DataFrame({"a": [3, 2], "b": [-1., -2]})
    train, val, test = TabularDataset.from_dfs(df1, val_df=df2, test_df=df3, fields={
        "a": CategoricalField(is_target=True),
        "b": [NumericField(normalization="Gaussian"), CategoricalField(handle_unk=True)],
    })
    train, val, test = TabularDataset.from_dfs(df1, val_df=df2, test_df=df3, fields={
        "a": CategoricalField(),
        "b": [NumericField(normalization="Gaussian", is_target=True), CategoricalField(handle_unk=True)],
    })

def test_real_data():
    df = pd.read_csv("./tests/resources/sample.csv")
    ds = TabularDataset.from_df(df, fields={
        "category_1": None,
        "category_3": None,
        "merchant_id": None,
        "subsector_id": CategoricalField(min_freq=3),
        "merchant_category_id": CategoricalField(min_freq=3),
        "city_id": None,
        "month_lag": NumericField(normalization="RankGaussian"),
        "card_id": None,
        "installments": NumericField(normalization=None),
        "state_id": CategoricalField(),
        "category_2": NumericField(normalization=None),
        "authorized_flag": CategoricalField(min_freq=3, handle_unk=True),
        "purchase_date": datetime_fields(),
        "purchase_amount": NumericField(normalization=None, fill_missing=None, is_target=True),
    }, train=True)

def test_real_data_csv():
    ds = TabularDataset.from_csv("./tests/resources/sample.csv", {
        "category_1": None,
        "category_3": None,
        "merchant_id": None,
        "subsector_id": CategoricalField(min_freq=3),
        "merchant_category_id": CategoricalField(min_freq=3),
        "city_id": None,
        "month_lag": NumericField(normalization="RankGaussian"),
        "card_id": None,
        "installments": NumericField(normalization=None),
        "state_id": CategoricalField(),
        "category_2": NumericField(normalization=None),
        "authorized_flag": CategoricalField(min_freq=3, handle_unk=True),
        "purchase_date": datetime_fields(),
        "purchase_amount": NumericField(normalization=None, fill_missing=None, is_target=True),
    }, train=True)

def test_multiple_cols():
    df1 = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    train = TabularDataset.from_df(df1, fields={
        "a": CategoricalField(is_target=True),
        "b": [NumericField(normalization="Gaussian"), CategoricalField(handle_unk=True)],
        ("a", "b"): Field(LambdaOperator(lambda x: x["a"] + x["b"])),
    })
    np.testing.assert_allclose(train.examples[("a", "b")].values, (df1["a"] + df1["b"]).values)

def test_fieldcollection():
    """Smoke test to confirm FieldCollection works with TabularDataset"""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": CategoricalField(max_features=100),
        "b": FieldCollection(NumericField(normalization="Gaussian"), Field(LambdaOperator(lambda x: x * 2))),
    })
    assert len(ds) == len(df)
    assert len(ds.fields) == 2
    for i in range(len(ds)):
        example = ds[i]
        assert "a" in example
        assert "b" in example
        assert len(example) == 2

def test_fieldcollection_flatten():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": CategoricalField(max_features=100),
        "b": FieldCollection(NumericField(normalization="Gaussian"), Field(LambdaOperator(lambda x: x * 2)), flatten=True),
    })
    assert len(ds) == len(df)
    assert len(ds.fields) == 3
    for i in range(len(ds)):
        example = ds[i]
        assert "a" in example
        assert "b" not in example
        assert len(example) == 3