import pytest
import pandas as pd

from torchtable.utils import *
from torchtable.operator import LambdaOperator
from torchtable.field import Field, FieldCollection, CategoricalField, NumericField
from torchtable.dataset import TabularDataset
from torchtable.loader import DefaultLoader
from torchtable.model import BatchHandlerModel

def test_from_dataset():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": CategoricalField(max_features=100),
        "b": NumericField(normalization="Gaussian"),
    })
    dl = DefaultLoader.from_dataset(ds, 5)
    model = BatchHandlerModel.from_dataset(ds)
    batch, _ = next(iter(dl))
    assert model(batch).size(0) == 5

def test_from_dataset_metadata():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5],
                       "c": [0.1, 0.2, 0.3, 0.4, 0.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": CategoricalField(max_features=100, metadata={"embedding_dim": 10}),
        "b": NumericField(normalization="Gaussian"),
        "c": NumericField(normalization=None),
    })
    dl = DefaultLoader.from_dataset(ds, 5)
    model = BatchHandlerModel.from_dataset(ds)
    batch, _ = next(iter(dl))
    assert model(batch).size(0) == 5
    assert model(batch).size(1) == 12
    assert model.out_dim() == 12

def test_from_dataset_field_collection():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": CategoricalField(max_features=100),
        "b": FieldCollection(NumericField(normalization="Gaussian"),
                             Field(LambdaOperator(lambda x: x * 2), continuous=True),
                             CategoricalField()),
    })
    dl = DefaultLoader.from_dataset(ds, 3)
    model = BatchHandlerModel.from_dataset(ds)
    assert len(model.embs) == 2
    batch, _ = next(iter(dl))
    assert model(batch).size(0) == 3

def test_from_dataset_only_categorical():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": CategoricalField(max_features=100),
        "b": None,
    })
    dl = DefaultLoader.from_dataset(ds, 4)
    model = BatchHandlerModel.from_dataset(ds)
    batch, _ = next(iter(dl))
    assert model(batch).size(0) == 4
    assert model.out_dim() == 15

def test_from_dataset_only_numerical():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": NumericField(),
        "b": FieldCollection(NumericField(), NumericField()),
    })
    dl = DefaultLoader.from_dataset(ds, 4)
    model = BatchHandlerModel.from_dataset(ds)
    batch, _ = next(iter(dl))
    assert model(batch).size(0) == 4
    assert model.out_dim() == 3

def test_from_dataset_flattened():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [-0.4, -2.1, 3.3, 4.4, 5.5]})
    ds = TabularDataset.from_df(df, fields={
        "a": NumericField(),
        "b": FieldCollection(NumericField(), CategoricalField(handle_unk=False), flatten=True),
    })
    dl = DefaultLoader.from_dataset(ds, 4)
    model = BatchHandlerModel.from_dataset(ds)
    batch, _ = next(iter(dl))
    assert model(batch).size(0) == 4