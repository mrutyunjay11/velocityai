class CommGroup:
    """
    Represents a communication group for distributed training.
    """
    def __init__(self, backend="nccl"):
        self.backend = backend
        self.rank = 0
        self.world_size = 1
        
    def all_reduce(self, tensor):
        """Reduces the tensor data across all machines."""
        # Stub: just returns the tensor for single-node execution
        pass
        
    def broadcast(self, tensor, src):
        """Broadcasts the tensor to all machines."""
        pass
