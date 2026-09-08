import math
import numpy as np
import velocityai as vai
from .dataset import Dataset

def default_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size."""
    elem = batch[0]
    if isinstance(elem, vai.Tensor):
        # We need a stack operation. Since we don't have vai.stack yet, we fallback to numpy
        np_arrays = [b.numpy() for b in batch]
        return vai.tensor(np.stack(np_arrays, axis=0))
    elif isinstance(elem, (int, float)):
        return vai.tensor(np.array(batch))
    elif isinstance(elem, tuple):
        transposed = zip(*batch)
        return tuple(default_collate(samples) for samples in transposed)
    raise TypeError(f"default_collate: batch must contain tensors, numbers, or tuples; found {type(elem)}")

class DataLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = default_collate

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.permutation(n) if self.shuffle else np.arange(n)
        
        batch = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
                
        if len(batch) > 0 and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        else:
            return math.ceil(n / self.batch_size)
