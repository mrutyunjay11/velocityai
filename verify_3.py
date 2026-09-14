import sys
import numpy as np
import time
import velocityai as vai
from velocityai.tokenizer import Tokenizer
from velocityai.generate import KVCache

def verify_model(model_id, prompt):
    print(f"\n--- Verifying {model_id} ---")
    model, config = vai.load_huggingface_model(model_id)
    tokenizer = Tokenizer(model_id)
    
    prompt_tokens = tokenizer.encode(prompt)
    
    # 1. C++ Batched Prefill
    from _velocityai_c import FastLlamaDecoder
    
    kv_cache = KVCache(
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        max_seq_len=len(prompt_tokens) + 128,
        head_dim=model.head_dim,
        device="cpu"
    )
    
    lm_head_c = model.embed_tokens.weight._c if (model.tie_word_embeddings or model.lm_head is None) else model.lm_head.weight._c
    fast_decoder = FastLlamaDecoder(
        model.vocab_size,
        model.dim,
        model.num_layers,
        model.num_heads,
        model.num_kv_heads,
        model.head_dim,
        model.intermediate_size,
        kv_cache.max_seq_len,
        model.rms_norm_eps,
        model.rope_theta,
        model.embed_tokens.weight._c,
        model.norm.weight._c,
        lm_head_c
    )
    for i, layer in enumerate(model.layers):
        fast_decoder.add_layer(
            layer.input_layernorm.weight._c,
            layer.self_attn.qkv_proj.weight._c,
            layer.self_attn.o_proj.weight._c,
            layer.post_attention_layernorm.weight._c,
            layer.mlp.gate_up_proj.weight._c,
            layer.mlp.down_proj.weight._c,
            kv_cache.k_tensors[i]._c,
            kv_cache.v_tensors[i]._c,
            layer.self_attn.qkv_proj.bias._c if layer.self_attn.qkv_proj.bias is not None else None
        )
        
    t0 = time.time()
    logits_tensor_cpp = vai.Tensor._wrap(fast_decoder.prefill_logits(prompt_tokens))
    t1 = time.time()
    prefill_time_cpp = t1 - t0
    print(f"C++ Prefill Time: {prefill_time_cpp:.4f}s")
    
    logits_cpp = logits_tensor_cpp.numpy()
    last_logits_cpp = logits_cpp[0, :] if logits_cpp.ndim == 2 else logits_cpp[:]
    
    top1_cpp = np.argmax(last_logits_cpp)
    top1_val_cpp = last_logits_cpp[top1_cpp]
    top5_cpp = set(np.argsort(last_logits_cpp)[-5:])
    
    # 2. Python Batched Prefill (Baseline)
    kv_cache_py = KVCache(
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        max_seq_len=len(prompt_tokens) + 128,
        head_dim=model.head_dim,
        device="cpu"
    )
    tokens_tensor = vai.tensor(np.array([prompt_tokens], dtype=np.int64)).to("cpu")
    
    t0 = time.time()
    logits_py = model(tokens_tensor, start_pos=0, last_token_only=True, kv_cache=kv_cache_py)
    t1 = time.time()
    prefill_time_py = t1 - t0
    print(f"Python Prefill Time: {prefill_time_py:.4f}s")
    
    logits_py_np = logits_py.numpy()
    last_logits_py = logits_py_np[0, -1, :] if logits_py_np.ndim == 3 else logits_py_np[-1, :]
    
    top1_py = np.argmax(last_logits_py)
    top1_val_py = last_logits_py[top1_py]
    top5_py = set(np.argsort(last_logits_py)[-5:])
    
    # Step 3c: Numerical Correctness Check
    print("\n--- Step 3c: Numerical Correctness (Logits) ---")
    print(f"Top-1 Match: {'PASS' if top1_cpp == top1_py else 'FAIL'} (C++: {top1_cpp}, Py: {top1_py})")
    
    top5_match = top5_cpp == top5_py
    print(f"Top-5 Unordered Match: {'PASS' if top5_match else 'FAIL'}")
    if not top5_match:
        print(f"  C++ Top 5: {top5_cpp}")
        print(f"  Py Top 5: {top5_py}")
        
    val_diff = abs(top1_val_cpp - top1_val_py)
    print(f"Top-1 Value Diff <= 0.1: {'PASS' if val_diff <= 0.1 else 'FAIL'} (Diff: {val_diff:.5f}, C++: {top1_val_cpp:.4f}, Py: {top1_val_py:.4f})")
    
    if top1_cpp != top1_py or not top5_match or val_diff > 0.1:
        print("-> Step 3c FAILED. Halting.")
        sys.exit(1)
        
    # Step 3d: Cache Continuation Check (10 decode steps)
    print("\n--- Step 3d: Cache Continuation (10 steps) ---")
    
    curr_token_cpp = top1_cpp
    curr_token_py = top1_py
    
    tokens_cpp = [curr_token_cpp]
    tokens_py = [curr_token_py]
    
    start_pos = len(prompt_tokens)
    
    for i in range(10):
        # C++ decode
        next_cpp = fast_decoder.decode_step(curr_token_cpp, start_pos + i)
        curr_token_cpp = next_cpp
        tokens_cpp.append(next_cpp)
        
        # Py decode
        input_tensor_py = vai.tensor(np.array([[curr_token_py]], dtype=np.int64)).to("cpu")
        logits_step_py = model(input_tensor_py, start_pos=start_pos + i, last_token_only=True, kv_cache=kv_cache_py)
        logits_step_py_np = logits_step_py.numpy()
        last_logits_step_py = logits_step_py_np[0, -1, :] if logits_step_py_np.ndim == 3 else logits_step_py_np[-1, :]
        next_py = np.argmax(last_logits_step_py)
        curr_token_py = next_py
        tokens_py.append(next_py)
        
    print(f"C++ Tokens: {tokens_cpp}")
    print(f"Py Tokens:  {tokens_py}")
    if tokens_cpp == tokens_py:
        print("Token-for-token match: PASS")
    else:
        print("Token-for-token match: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    prompt = "Explain why hibernating bears don't need to eat, drink, or urinate for months. Keep the answer to 4-5 sentences, final answer only."
    verify_model("HuggingFaceTB/SmolLM2-360M-Instruct", prompt)
    verify_model("Qwen/Qwen2.5-1.5B-Instruct", prompt)
