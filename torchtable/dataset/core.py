import logging
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from ..custom_types import *

from ..operator import Operator, LambdaOperator, FillMissing, Categorize, Normalize

from ..field import Field

logger = logging.getLogger(__name__)

class TabularDataset(torch.utils.data.Dataset):
    
    def __init__(self, examples: Dict[ColumnName, ArrayLike],
                 fields: Dict[ColumnName, Field], train=True):
        self.examples = examples
        # all fields should be of the same length
        self.length = len(next(iter(self.examples.values())))
        self.fields = fields
        self.train = train
        self.continuous_cols = [fld.name for fld in fields.values() if fld.continuous and not fld.is_target]
        self.categorical_cols = [fld.name for fld in fields.values() if fld.categorical and not fld.is_target]
        self.target_cols = [fld.name for fld in fields.values() if fld.is_target]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx) -> Dict[str, ArrayLike]:
        return {k: v[idx] for k, v in self.examples.items()}
    
    def __repr__(self):
        fields_rep = ",\n".join([" " * 4 + str(x) for x in self.fields.values()])
        nl = "\n"
        return f"TabularDataset({nl + fields_rep + nl})"
        
    @classmethod
    def from_df(cls, df: pd.DataFrame, fields: Dict[ColumnName, OneorMore[Field]],
                train=True) -> 'TabularDataset':
        missing_cols = set(df.columns) - set(fields.keys())
        if len(missing_cols) > 0:
            logger.warning(f"The following columns are missing from the fields list: {missing_cols}")
        
        field_to_source = {}
        for k, fld in fields.items():
            if fld is None: continue
            if isinstance(fld, (tuple, list)):
                # if multiple fields are specified, hook them all to the same column
                for i, f in enumerate(fld):
                    field_to_source[f] = df[k]
                    if f.name is None: f.name = f"{k}_{i}"
            else:
                field_to_source[fld] = df[k]
                if fld.name is None: fld.name = k
        
        examples = {}
        for fld, src in field_to_source.items():
            if fld.name in examples:
                logger.warning("Some fields have duplicate names. This will cause previous fields to be overwritten.")
            examples[fld.name] = fld.transform(src, train=train)
        return cls(examples, {fld.name: fld for fld in field_to_source.keys()}, train=train)
    
    @classmethod
    def from_dfs(cls, train_df: pd.DataFrame, 
                 val_df: pd.DataFrame=None, test_df: pd.DataFrame=None,
                 fields: Dict[ColumnName, OneorMore[Field]]=None) -> Iterable['TabularDataset']:
        train = cls.from_df(train_df, fields, train=True)
        yield train
        if val_df is not None:
            yield cls.from_df(val_df, fields, train=False)
        if test_df is not None:
            # remove all target fields
            non_target_fields = {}
            for k, fld in fields.items():
                if fld is None: continue
                if isinstance(fld, (tuple, list)):
                    non_target_fields[k] = []
                    for f in fld:
                        if not f.is_target: non_target_fields[k].append(f)
                    if len(non_target_fields[k]) == 0: non_target_fields[k] = None
                else:
                    if not fld.is_target:
                        non_target_fields[k] = fld
                    else:
                        non_target_fields[k] = None
            yield cls.from_df(test_df, non_target_fields, train=False)
        
    @classmethod
    def from_csv(cls, fname: str, fields: Dict[ColumnName, OneorMore[Field]],
                 train=True, csv_read_params: dict={}) -> 'TabularDataset':
        return cls.from_df(pd.read_csv(fname, **csv_read_params), fields=fields, train=train)