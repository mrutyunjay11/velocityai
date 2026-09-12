import numpy as np
import velocityai as vai
from velocityai.nn.module import Module
from velocityai.nn.linear import Linear
from velocityai.nn.rope import RotaryEmbedding, apply_rotary_pos_emb

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_weights: bool = True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if init_weights:
            w = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
            self.weight = vai.tensor(w)
            self.weight.requires_grad = True
            self._parameters["weight"] = self.weight
        else:
            self.weight = None

    def forward(self, x):
        if isinstance(x, vai.Tensor):
            x_np = x.numpy().astype(np.int64)
        elif isinstance(x, np.ndarray):
            x_np = x.astype(np.int64)
        else:
            x_np = np.array(x, dtype=np.int64)
            
        w_np = self.weight.numpy()
        out_np = w_np[x_np]
        out_tensor = vai.tensor(out_np.astype(np.float32))
        dev = self.weight.device
        if dev != "cpu":
            out_tensor = out_tensor.to(dev)
        return out_tensor

class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = vai.ones((dim,))
        self.weight.requires_grad = True
        self._parameters["weight"] = self.weight

    def forward(self, x):
        from _velocityai_c import rms_norm
        return vai.Tensor._wrap(rms_norm(x._c, self.weight._c, self.eps))

class CausalSelfAttention(Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int = None, head_dim: int = None, rope_theta: float = 100000.0, max_seq_len: int = 4096, attention_bias: bool = False, init_weights: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else (dim // num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = Linear(dim, self.num_heads * self.head_dim, bias=attention_bias, init_weights=init_weights)
        self.k_proj = Linear(dim, self.num_kv_heads * self.head_dim, bias=attention_bias, init_weights=init_weights)
        self.v_proj = Linear(dim, self.num_kv_heads * self.head_dim, bias=attention_bias, init_weights=init_weights)
        self.o_proj = Linear(self.num_heads * self.head_dim, dim, bias=False, init_weights=init_weights)
        self.qkv_proj = None
        
        self.rotary = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, base=rope_theta)

    def forward(self, x, start_pos: int = 0, kv_cache=None, layer_idx=None):
        # x: (B, T, dim)
        B, T, _ = x.shape
        
        if self.qkv_proj is not None:
            qkv = self.qkv_proj(x)
            q_dim = self.num_heads * self.head_dim
            k_dim = self.num_kv_heads * self.head_dim
            q = vai.Tensor._wrap(qkv._c.slice(2, 0, q_dim))
            k = vai.Tensor._wrap(qkv._c.slice(2, q_dim, q_dim + k_dim))
            v = vai.Tensor._wrap(qkv._c.slice(2, q_dim + k_dim, q_dim + 2 * k_dim))
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
        # Fast path for single-token decode (T=1) using fused C++ SIMD kernel
        if T == 1 and kv_cache is not None and layer_idx is not None and hasattr(kv_cache, 'k_tensors'):
            from _velocityai_c import fused_attention_decode
            cos, sin = self.rotary.get_cos_sin(seq_len=1, offset=start_pos)
            cos_t = vai.from_numpy(np.ascontiguousarray(cos[0], dtype=np.float32), copy=False)
            sin_t = vai.from_numpy(np.ascontiguousarray(sin[0], dtype=np.float32), copy=False)
            
            context = fused_attention_decode(
                q._c, k._c, v._c,
                kv_cache.k_tensors[layer_idx]._c,
                kv_cache.v_tensors[layer_idx]._c,
                cos_t._c, sin_t._c,
                self.num_heads, self.num_kv_heads, self.head_dim,
                kv_cache.max_seq_len, start_pos
            )
            return self.o_proj(vai.Tensor._wrap(context))
        
        # Reshape to (B, num_heads, T, head_dim)
        q_np = q.numpy().reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k_np = k.numpy().reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v_np = v.numpy().reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        cos, sin = self.rotary.get_cos_sin(seq_len=T, offset=start_pos)
        q_np, k_np = apply_rotary_pos_emb(q_np, k_np, cos, sin)
        
        # Cache update
        if kv_cache is not None and layer_idx is not None:
            k_np, v_np = kv_cache.update(layer_idx, k_np, v_np, start_pos)
            
        # Repeat KV heads for GQA if num_kv_groups > 1
        if self.num_kv_groups > 1:
            k_np = np.repeat(k_np, self.num_kv_groups, axis=1)
            v_np = np.repeat(v_np, self.num_kv_groups, axis=1)
            
        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q_np @ k_np.transpose(0, 1, 3, 2)) * scale
        
        # Causal masking if sequence length > 1
        if T > 1:
            Total_T = k_np.shape[2]
            mask = np.tril(np.ones((T, Total_T), dtype=np.float32), k=(Total_T - T))
            scores = np.where(mask == 0, -1e9, scores)
            
        # Softmax along last dim
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        att_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Weighted sum over V
        context = att_weights @ v_np
        context = context.transpose(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)
        
        return self.o_proj(vai.tensor(context.astype(np.float32)))

class SwiGLUMLP(Module):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))"""
    def __init__(self, dim: int, intermediate_size: int, init_weights: bool = True):
        super().__init__()
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.gate_proj = Linear(dim, intermediate_size, bias=False, init_weights=init_weights)
        self.up_proj = Linear(dim, intermediate_size, bias=False, init_weights=init_weights)
        self.down_proj = Linear(intermediate_size, dim, bias=False, init_weights=init_weights)
        self.gate_up_proj = None

    def forward(self, x):
        is_single_token = (x.ndim == 3 and x.shape[1] == 1) or (x.ndim == 2 and x.shape[0] == 1)
        if self.gate_up_proj is not None and is_single_token:
            from _velocityai_c import gemv_swiglu
            hidden = vai.Tensor._wrap(gemv_swiglu(x._c, self.gate_up_proj.weight._c))
            return self.down_proj(hidden)

        from _velocityai_c import swiglu
        if self.gate_up_proj is not None:
            gate_up = self.gate_up_proj(x)
            gate = vai.Tensor._wrap(gate_up._c.slice(2, 0, self.intermediate_size))
            up = vai.Tensor._wrap(gate_up._c.slice(2, self.intermediate_size, 2 * self.intermediate_size))
            hidden = vai.Tensor._wrap(swiglu(gate._c, up._c))
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            hidden = vai.Tensor._wrap(swiglu(gate._c, up._c))
        return self.down_proj(hidden)

class TransformerBlock(Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, intermediate_size: int, rms_norm_eps: float = 1e-5, rope_theta: float = 100000.0, attention_bias: bool = False, init_weights: bool = True):
        super().__init__()
        self.self_attn = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            init_weights=init_weights
        )
        self.mlp = SwiGLUMLP(dim=dim, intermediate_size=intermediate_size, init_weights=init_weights)
        self.input_layernorm = RMSNorm(dim, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(dim, eps=rms_norm_eps)

    def forward(self, x, start_pos: int = 0, kv_cache=None, layer_idx=None):
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attn(norm_x, start_pos=start_pos, kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + attn_out
        
        norm_x2 = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x2)
        x = x + mlp_out
        return x

class LlamaModel(Module):
    """Full Llama-style model architecture matching HuggingFace config."""
    def __init__(self, config: dict = None, **kwargs):
        super().__init__()
        cfg = dict(config or {})
        cfg.update(kwargs)
        
        self.vocab_size = cfg.get("vocab_size", 49152)
        self.dim = cfg.get("hidden_size", cfg.get("dim", 960))
        self.num_layers = cfg.get("num_hidden_layers", cfg.get("num_layers", 32))
        self.num_heads = cfg.get("num_attention_heads", cfg.get("num_heads", 15))
        self.num_kv_heads = cfg.get("num_key_value_heads", cfg.get("num_kv_heads", 5))
        self.head_dim = cfg.get("head_dim", self.dim // self.num_heads)
        self.intermediate_size = cfg.get("intermediate_size", 2560)
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-5)
        self.rope_theta = cfg.get("rope_theta", 100000.0)
        self.tie_word_embeddings = cfg.get("tie_word_embeddings", True)
        self.attention_bias = cfg.get("attention_bias", False)
        
        init_weights = cfg.get("init_weights", kwargs.get("init_weights", True))
        self.embed_tokens = Embedding(self.vocab_size, self.dim, init_weights=init_weights)
        
        self.layers = []
        for i in range(self.num_layers):
            block = TransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                intermediate_size=self.intermediate_size,
                rms_norm_eps=self.rms_norm_eps,
                rope_theta=self.rope_theta,
                attention_bias=self.attention_bias,
                init_weights=init_weights
            )
            self.layers.append(block)
            self._modules[f"layer_{i}"] = block
            
        self.norm = RMSNorm(self.dim, eps=self.rms_norm_eps)
        
        if not self.tie_word_embeddings:
            self.lm_head = Linear(self.dim, self.vocab_size, bias=False, init_weights=init_weights)
        else:
            self.lm_head = None

    def forward(self, input_ids, start_pos: int = 0, last_token_only: bool = False, kv_cache=None):
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos=start_pos, kv_cache=kv_cache, layer_idx=i)
        h = self.norm(h)
        
        if last_token_only:
            # Sliced in C++ without NumPy copy
            T_len = h.shape[1]
            flat_h = vai.Tensor._wrap(h._c.slice(1, T_len - 1, T_len)).reshape(1, self.dim)
        else:
            flat_h = h.reshape(-1, self.dim)
            
        if self.tie_word_embeddings or self.lm_head is None:
            from _velocityai_c import linear
            return vai.Tensor._wrap(linear(flat_h._c, self.embed_tokens.weight._c))
        else:
            return self.lm_head(flat_h)
