import numpy as np
import velocityai as vai
from velocityai.nn.rope import RotaryEmbedding, apply_rotary_pos_emb
from velocityai.nn.transformer import LlamaModel, CausalSelfAttention, SwiGLUMLP

def test_rope():
    print("Testing RoPE...")
    head_dim = 64
    seq_len = 16
    rotary = RotaryEmbedding(dim=head_dim, max_seq_len=128, base=100000.0)
    cos, sin = rotary.get_cos_sin(seq_len=seq_len)
    
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)
    
    q = np.random.randn(1, 15, seq_len, head_dim).astype(np.float32)
    k = np.random.randn(1, 5, seq_len, head_dim).astype(np.float32)
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print("RoPE passed!")

def test_gqa_attention():
    print("Testing GQA Attention...")
    dim = 960
    num_heads = 15
    num_kv_heads = 5
    head_dim = 64
    
    attn = CausalSelfAttention(dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
    
    B, T = 1, 8
    x = vai.randn(B, T, dim)
    out = attn(x)
    assert out.shape == (B, T, dim)
    print("GQA Attention passed!")

def test_swiglu_mlp():
    print("Testing SwiGLU MLP...")
    dim = 960
    intermediate_size = 2560
    mlp = SwiGLUMLP(dim=dim, intermediate_size=intermediate_size)
    
    x = vai.randn(1, 4, dim)
    out = mlp(x)
    assert out.shape == (1, 4, dim)
    print("SwiGLU MLP passed!")

def test_smollm2_architecture():
    print("Testing Full SmolLM2 Architecture (2-layer mini test)...")
    config = {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "tie_word_embeddings": True
    }
    model = LlamaModel(config=config)
    
    input_ids = np.array([[12, 45, 99, 102]], dtype=np.int64)
    logits = model(input_ids)
    assert logits.shape == (1, 4, 1000)
    print("SmolLM2 Architecture test passed!")

if __name__ == "__main__":
    test_rope()
    test_gqa_attention()
    test_swiglu_mlp()
    test_smollm2_architecture()
    print("ALL TESTS PASSED SUCCESSFULLY!")
