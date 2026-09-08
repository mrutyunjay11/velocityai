import velocityai as vai
import numpy as np

class KVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, max_seq_len: int, head_dim: int, device: str = "cpu", dtype=np.float32):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        
        # Pre-allocate cache arrays (1, num_kv_heads, max_seq_len, head_dim)
        self.k = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=dtype) for _ in range(num_layers)]
        self.v = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=dtype) for _ in range(num_layers)]
        
        # C++ tensors for fused decode attention
        self.k_tensors = [vai.zeros(max_seq_len, num_kv_heads, head_dim) for _ in range(num_layers)]
        self.v_tensors = [vai.zeros(max_seq_len, num_kv_heads, head_dim) for _ in range(num_layers)]
        
    def update(self, layer_idx, new_k, new_v, start_pos):
        """
        Inserts new_k and new_v into the cache at start_pos.
        new_k, new_v shapes: (B, num_kv_heads, T, head_dim)
        Returns the full concatenated k, v up to start_pos + T.
        """
        T = new_k.shape[2]
        
        self.k[layer_idx][:, :, start_pos:start_pos + T, :] = new_k
        self.v[layer_idx][:, :, start_pos:start_pos + T, :] = new_v
        
        # Sync to C++ tensors for prefill tokens
        k_slice = new_k[0].transpose(1, 0, 2) # (T, num_kv_heads, head_dim)
        v_slice = new_v[0].transpose(1, 0, 2)
        k_t_np = self.k_tensors[layer_idx].numpy()
        v_t_np = self.v_tensors[layer_idx].numpy()
        k_t_np[start_pos:start_pos + T] = k_slice
        v_t_np[start_pos:start_pos + T] = v_slice
        
        return (self.k[layer_idx][:, :, :start_pos + T, :],
                self.v[layer_idx][:, :, :start_pos + T, :])
