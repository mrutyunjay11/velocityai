import numpy as np

class RotaryEmbedding:
    """Rotary Position Embedding (RoPE) as used in Llama, SmolLM2, and Mistral."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 100000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # inv_freq: (dim // 2,)
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2, dtype=np.float32) / self.dim))
        self.inv_freq = inv_freq
        
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq) # (max_seq_len, dim // 2)
        
        # Concat to match head_dim: (max_seq_len, dim)
        emb = np.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = np.cos(emb).astype(np.float32)
        self.sin_cached = np.sin(emb).astype(np.float32)

    def get_cos_sin(self, seq_len: int, offset: int = 0):
        """Returns cos and sin for positions [offset, offset + seq_len)."""
        end = offset + seq_len
        if end > self.max_seq_len:
            # Dynamically extend cache if sequence length exceeds initial bound
            t = np.arange(end, dtype=np.float32)
            freqs = np.outer(t, self.inv_freq)
            emb = np.concatenate((freqs, freqs), axis=-1)
            self.cos_cached = np.cos(emb).astype(np.float32)
            self.sin_cached = np.sin(emb).astype(np.float32)
            self.max_seq_len = end
            
        cos = self.cos_cached[offset:end]
        sin = self.sin_cached[offset:end]
        return cos, sin

def rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotates half the hidden dims of the input."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return np.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Applies RoPE to Q and K.
    q: (B, num_q_heads, T, head_dim)
    k: (B, num_kv_heads, T, head_dim)
    cos, sin: (T, head_dim) or (1, 1, T, head_dim)
    """
    if cos.ndim == 2:
        cos = np.expand_dims(cos, axis=(0, 1)) # (1, 1, T, head_dim)
        sin = np.expand_dims(sin, axis=(0, 1))
        
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot.astype(np.float32), k_rot.astype(np.float32)
