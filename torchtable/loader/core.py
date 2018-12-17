import math
import random
import numpy as np
import pandas as pd
from contextlib import contextmanager
from copy import deepcopy

import torch.utils.data

from ..custom_types import *

from ..utils import *

class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic. Borrowed from torchtext."""

    def __init__(self, random_state=None):
        self._random_state = random_state
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))

ProcessedBatch = Tuple[Dict[str, torch.tensor], Dict[str, torch.tensor]]

class DefaultLoader(torch.utils.data.DataLoader):
    """Defines an iterator that loads batches of data from a Dataset.
    Heavily based on the Iterator from torchtext.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        repeat: Whether to repeat the iterator for multiple epochs. Default: False.
        shuffle: Whether to shuffle examples between epochs.
        device (str or `torch.device`): A string or instance of `torch.device`
            specifying which device the Variables are going to be created on.
            If None, the tensors will be created on cpu. Default: None.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int,
                 device: Optional[torch.device]=None, repeat: bool=False,
                 shuffle: Optional[bool]=None):
        self.batch_size, self.dataset = batch_size, dataset
        self.iterations = 0
        self.repeat = repeat
        self.shuffle = with_default(shuffle, self.dataset.train)

        if isinstance(device, int):
            warnings.warn("The `device` argument should be set by using `torch.device`" +
                           " or passing a string as an argument. This behavior will be" +
                           " deprecated soon and currently defaults to cpu.")
            device = None
        self.device = device
        if self.shuffle:
            # TODO: Clean interface
            self.index_generator = RandomShuffler()
        else:
            self.index_generator = lambda x: x

        # For state loading/saving only
        self._iterations_this_epoch = 0
        self._random_state_this_epoch = None
        self._restored_from_state = False
    
    @classmethod
    def from_dataset(cls, dataset: torch.utils.data.Dataset, batch_size: int,
                 device: torch.device=None, repeat: bool=False, shuffle: Optional[bool]=None):
        return cls(dataset, batch_size, device=device, repeat=repeat, shuffle=shuffle)
    
    @classmethod
    def from_datasets(cls, train_ds: torch.utils.data.Dataset, batch_size: OneorMore[int],
                      val_ds: Optional[torch.utils.data.Dataset]=None, test_ds: Optional[torch.utils.data.Dataset]=None,
                      device: OneorMore[torch.device]=None, repeat: OneorMore[bool]=False,
                      shuffle: Optional[OneorMore[Optional[bool]]]=None) -> Iterable['DefaultLoader']:
        n_ds = 1
        if val_ds is not None: n_ds += 1
        if test_ds is not None: n_ds += 1
            
        args = (expand(batch_size, n_ds), )
        kwargs = {
            "device": expand(device, n_ds),
            "repeat": expand(repeat, n_ds),
            "shuffle": expand(shuffle, n_ds),
        }
        
        i = 0
        yield cls.from_dataset(train_ds, *([a[i] for a in args]), **({k: v[i] for k, v in kwargs.items()}))
        i += 1
        if val_ds is not None:
            yield cls.from_dataset(val_ds, *([a[i] for a in args]), **({k: v[i] for k, v in kwargs.items()}))
            i += 1
        if test_ds is not None:
            yield cls.from_dataset(test_ds, *([a[i] for a in args]), **({k: v[i] for k, v in kwargs.items()}))

    def _process_batch(self, data: Dict[str, ArrayLike]) -> ProcessedBatch:
        """Converts examples in a dataset to model inputs by using the fields to transform
        the inputs to tensors. Implement in subclass to add custom behavior."""
        in_data = {}
        tgt_data = {}
        for k, v in data.items():
            fld = self.dataset.fields[k]
            tsr = fld.to_tensor(v, device=self.device, train=self.dataset.train)
            if fld.is_target: tgt_data[k] = tsr
            else: in_data[k] = tsr
        return in_data, tgt_data
            
    def _batches(self) -> Iterable[ProcessedBatch]:
        """Iterates through the dataset while generating batches of input and target variables.
        Assumes dataset can be indexed using a list."""
        indices = []
        for i in self.index_generator(range(len(self.dataset))):
            indices.append(i)
            if len(indices) == self.batch_size:
                yield self._process_batch(self.dataset[indices])
                indices = []
        if len(indices) > 0:
            yield self._process_batch(self.dataset[indices])    

    def init_epoch(self):
        """Set up the batch generator for a new epoch."""
        if self.shuffle:
            if self._restored_from_state:
                self.index_generator.random_state = self._random_state_this_epoch
            else:
                self._random_state_this_epoch = self.index_generator.random_state
        
        if self._restored_from_state:
            self._restored_from_state = False
        else:
            self._iterations_this_epoch = 0

        if not self.repeat: self.iterations = 0
    
    @property
    def epoch(self):
        return math.floor(self.iterations / len(self))

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterable[Dict[str, torch.tensor]]:
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self._batches()):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                yield minibatch
            if not self.repeat:
                break

    def state_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "iterations_this_epoch": self._iterations_this_epoch,
            "random_state_this_epoch": self._random_state_this_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.iterations = state_dict["iterations"]
        self._iterations_this_epoch = state_dict["iterations_this_epoch"]
        self._random_state_this_epoch = state_dict["random_state_this_epoch"]
        self._restored_from_state = True

class ContinuousJoinLoader(DefaultLoader):
    """Joins all continuous input fields into a single output field."""
    def _process_batch(self, data: Dict[str, ArrayLike]) -> ProcessedBatch:
        x, y = super()._process_batch(data)
        x_cont_keys = [k for k in x.keys() if self.dataset.fields[k].continuous]
        x["continous"] = torch.cat([x.pop(k).reshape((-1, 1)) for k in x_cont_keys], dim=1)
        return x, y