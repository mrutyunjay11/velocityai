class Dataset:
    """An abstract class representing a Dataset."""
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class TensorDataset(Dataset):
    """Dataset wrapping tensors.
    
    Each sample will be retrieved by indexing tensors along the first dimension.
    """
    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]
