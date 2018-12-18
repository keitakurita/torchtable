import pytest
import itertools

from torchtable import *
from torchtable.field import *
from torchtable.dataset import *
from torchtable.loader import *

def test_from_dataset():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": CategoricalField(max_features=100),
        "b": [NumericField(normalization="Gaussian"), IdentityField()],
    })
    dl = DefaultLoader

def test_from_datasets():
    df1 = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    df2 = pd.DataFrame({"a": [1, 2, 3], "b": [-1., -2, -3.]})
    df3 = pd.DataFrame({"a": [3, 2], "b": [-1., -2]})
    train, val, test = TabularDataset.from_dfs(df1, val_df=df2, test_df=df3, fields={
        "a": CategoricalField(),
        "b": [NumericField(normalization="Gaussian"), CategoricalField(handle_unk=True)],
    })
    # all present
    train_dl, val_dl, test_dl = DefaultLoader.from_datasets(train, 3, val_ds=val, test_ds=test)
    # val only
    train_dl, val_dl = DefaultLoader.from_datasets(train, 3, val_ds=val, test_ds=None)
    # test only
    train_dl, test_dl = DefaultLoader.from_datasets(train, 3, val_ds=None, test_ds=test)

def test_from_datasets_multiple_args():
    df1 = pd.DataFrame({"a": [3, 4, 5, 1, 2],
                       "b": [1.3, -2.1, 2.3, 5.4, 5.6]})
    df2 = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [-1., -2, -3., -4., -5.]})
    df3 = pd.DataFrame({"a": [3, 2], "b": [-1., -2]})
    train, val, test = TabularDataset.from_dfs(df1, val_df=df2, test_df=df3, fields={
        "a": CategoricalField(),
        "b": [NumericField(normalization="Gaussian"), CategoricalField(handle_unk=True)],
    })
    train_dl, val_dl, test_dl = DefaultLoader.from_datasets(train, (5, 3, 2), val_ds=val, test_ds=test,
                                                            device=(None, None, None), repeat=(True, True, True),
                                                            shuffle=(True, True, True))
    x, y = next(iter(train_dl))
    for v in itertools.chain(x.values(), y.values()): assert v.size()[0] == 5
    x, y = next(iter(val_dl))
    for v in itertools.chain(x.values(), y.values()): assert v.size()[0] == 3
    x, y = next(iter(test_dl))
    for v in itertools.chain(x.values(), y.values()): assert v.size()[0] == 2
        
    train_dl, val_dl = DefaultLoader.from_datasets(train, (3, 4), val_ds=val, test_ds=None)
    x, y = next(iter(train_dl))
    for v in itertools.chain(x.values(), y.values()): assert v.size()[0] == 3
    x, y = next(iter(val_dl))
    for v in itertools.chain(x.values(), y.values()): assert v.size()[0] == 4

def test_real_data():
    """Smoke test for real dataset"""
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
    
    bs = 32
    x, y = next(iter(DefaultLoader.from_dataset(ds, bs)))
    for v in itertools.chain(x.values(), y.values()):
        assert v.size()[0] == bs

def test_continuous_join_loader():
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
    
    bs = 32
    x, y = next(iter(ContinuousJoinLoader.from_dataset(ds, bs)))
    for v in itertools.chain(x.values(), y.values()):
        assert v.size()[0] == bs