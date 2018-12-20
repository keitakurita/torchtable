import math
import numpy as np
import pandas as pd

from ..custom_types import *

from torchtable.utils import *
from torchtable.field import Field, FieldCollection, CategoricalField, NumericField
from torchtable.dataset import TabularDataset

import torch
import torch.nn as nn

class BatchHandlerModel(nn.Module):
    def __init__(self, embs: List[nn.Module],
                 batch_cat_field_getters: List[Callable[[Dict], torch.tensor]],
                 batch_num_field_getters: Callable[[Dict], torch.tensor]):
        super().__init__()
        assert len(embs) == len(batch_cat_field_getters)
        self.embs = nn.ModuleList(embs)
        self.batch_cat_field_getters = batch_cat_field_getters
        self.batch_num_field_getters = batch_num_field_getters
    
    @staticmethod
    def field_to_embedding(fld: CategoricalField) -> nn.Module:
        num_embeddings = fld.cardinality
        embedding_dim = with_default(fld.metadata.get("embedding_dim"),
                                      min((num_embeddings * (num_embeddings - 1)) // 2, 50))
        return nn.Embedding(num_embeddings, embedding_dim, padding_idx=fld.metadata.get("padding_idx"))
    
    @classmethod
    def from_dataset(cls, dataset: TabularDataset) -> 'DefaultModel':
        # construct mapping from example field to embedding matrix
        embs = []
        batch_cat_field_getters: List[Callable[[Dict], torch.tensor]] = []
        batch_num_field_getters: List[Callable[[Dict], torch.tensor]] = []
        def register_field(k: str, fld: Field, i: int):
            getter = (lambda b: b[k][i]) if i > -1 else (lambda b: b[k])
            if fld.categorical:
                embs.append(cls.field_to_embedding(fld))
                batch_cat_field_getters.append(getter)
            elif fld.continuous:
                batch_num_field_getters.append(getter)
        list(dataset.fields.flatmap(register_field, with_index=True))
        return cls(embs, batch_cat_field_getters, batch_num_field_getters)
    
    def forward(self, batch):
        cat_data = [emb(getter(batch)) for emb, getter in zip(self.embs, self.batch_cat_field_getters)]
        num_data = [getter(batch).unsqueeze(1) for getter in self.batch_num_field_getters]
        return torch.cat(cat_data + num_data, dim=1)

    def out_dim(self):
        return sum(e.embedding_dim for e in self.embs) + len(self.batch_num_field_getters)