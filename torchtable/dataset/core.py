import logging
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from ..custom_types import *

from ..utils import with_default, flat_filter, apply_oneormore, fold_oneormore
from ..operator import Operator, LambdaOperator, FillMissing, Categorize, Normalize
from ..field import *

logger = logging.getLogger(__name__)

FieldOrFields = Union[Field, FieldCollection, Collection[Field]]

class TabularDataset(torch.utils.data.Dataset):
    """
    A dataset for tabular data.
    Args:
        fields: A dictionary mapping from a column/columns in the raw data to a Field/Fields.
                To specify multiple columns as input, use a tuple of column names.
                To map a single column to multiple fields, use a list of fields.
                Each field will be mapped to a single entry in the processed dataset.
        train: Whether this dataset is the training set. This affects whether the fields will fit the given data.
    """
    def __init__(self, examples: Dict[ColumnName, OneorMore[ArrayLike]],
                 fields: Dict[ColumnName, Union[Field, FieldCollection]], train=True):
        self.examples = examples
        example = next(iter(self.examples.values()))
        self.length = fold_oneormore(lambda x,y: len(y), example, [])
        self.fields = fields
        self.train = train
        self.continuous_fields = list(flat_filter(fields.values(), lambda x: x.continuous and not x.is_target))
        self.categorical_fields = list(flat_filter(fields.values(), lambda x: x.categorical and not x.is_target))
        self.target_fields = list(flat_filter(fields.values(), lambda x: x.is_target))
                                  
    def __len__(self):
        return self.length
    
    def _index_example(self, k: ColumnName, val: OneorMore[ArrayLike], idx) -> OneorMore[ArrayLike]:
        # check if the field is a tuple/list since the field output might be a list
        # even though the field itself is not
        if isinstance(self.fields[k], (tuple, list)):
            return [v[idx] for v in val]
        else:
            return val[idx]
    
    def __getitem__(self, idx) -> Dict[str, ArrayLike]:
        return {k: self.fields[k].index(v, idx) for k, v in self.examples.items()}
    
    def __repr__(self):
        fields_rep = ",\n".join([" " * 4 + str(x) for x in self.fields.values()])
        nl = "\n"
        return f"TabularDataset({nl + fields_rep + nl})"
        
    @classmethod
    def from_df(cls, df: pd.DataFrame, fields: Dict[ColumnName, FieldOrFields],
                train=True) -> 'TabularDataset':
        """
        Initialize a dataset from a pandas dataframe.
        Args:
            df: pandas dataframe to initialize from
            fields: Dictionary mapping from a column identifier to a field or fields.
            The key can be a single column name or a tuple of multiple columns. The column(s) specified by the key will be passed to the field(s) transform method.
            The value can be a single field, a list/tuple of fields, or a `field.FieldCollection`.
            In general, each example in the dataset will mirror the structure of the fields passed.
            For instance, if you pass multiple fields for a certain key, the example will also have multiple outputs for the given key structured as a list.
        Example:
        >>> df.head(2)
                  authorized_flag          card_id  price
        0               Y  C_ID_4e6213e9bc       1.2
        1               Y  C_ID_4e6213e9bc       3.4
        >>> ds = TabularDataset.from_df(df, fields={
        ...     "authorized_flag": CategoricalField(handle_unk=False), # standard field
        ...     "card_id": [CategoricalField(handle_unk=True),
        ...                 Field(LambdaOperator(lambda x: x.str[0]) > Categorize())], # multiple fields and custom fields
        ...     "price": NumericField(fill_missing=None, normalization=None, is_target=True), # target field
        ...     ("authorized_flag", "price"): Field(LambdaOperator(
        ...             lambda x: (x["authorized_flag"] == "N").astype("int") * x["price"])), # multiple column field
        ... })
        >>> ds[0] 
        {"authorized_flag": 0,
         "card_id": [1, 0],
          "price": 1.2,
          ("authorized_flag", "price"): 0.}
        """
        missing_cols = set(df.columns) - set(fields.keys())
        if len(missing_cols) > 0:
            logger.warning(f"The following columns are missing from the fields list: {missing_cols}")
        
        # convert raw lists/tuples of fields to FieldCollections
        for k, v in fields.items():
            if type(v) in (tuple, list): fields[k] = FieldCollection(*v)
        
        def _to_df_key(k):
            # change tuples to lists for accessing dataframe columns
            # this is necessary since lists cannot be dictionary keys
            if isinstance(k, tuple): return list(k)
            else: return k

        # construct examples while setting names
        examples = {}            
        for k, fld in fields.items():
            if fld is None: continue
            # fields are either a Field or FieldCollection, so the following code works
            fld.name = k
            examples[k] = fld.transform(df[_to_df_key(k)], train=train)
        
        return cls(examples, {k: v for k, v in fields.items() if v is not None}, train=train)
    
    @classmethod
    def from_dfs(cls, train_df: pd.DataFrame, 
                 val_df: pd.DataFrame=None, test_df: pd.DataFrame=None,
                 fields: Dict[ColumnName, OneorMore[Field]]=None) -> Iterable['TabularDataset']:
        """
        Generates datasets from train, val, and test dataframes.
        Example:
        >>> trn, val, test = TabularDataset.from_dfs(train_df, val_df=val_df, test_df=test_df, fields={
        ...   "a": NumericalField(), "b": CategoricalField(),
        ...  })
        """
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
        """
        Initialize a dataset from a csv file.
        Kwargs:
            csv_read_params: Keyword arguments to pass to the `pd.read_csv` method.
        """
        return cls.from_df(pd.read_csv(fname, **csv_read_params), fields=fields, train=train)