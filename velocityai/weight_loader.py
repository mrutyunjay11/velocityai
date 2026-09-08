import os
import json
import numpy as np
from safetensors.numpy import load_file
import velocityai as vai
from velocityai.nn.transformer import LlamaModel
from velocityai.nn.linear import Linear

def load_huggingface_model(model_dir_or_id: str, device: str = "cpu") -> tuple[LlamaModel, dict]:
    """
    Loads a Llama/SmolLM model from a local directory or downloads from HuggingFace Hub.
    Returns (model, config).
    """
    model_dir = model_dir_or_id
    if not os.path.exists(model_dir):
        from huggingface_hub import snapshot_download
        try:
            # Try offline first for instant load
            model_dir = snapshot_download(
                repo_id=model_dir_or_id,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
                local_files_only=True
            )
        except Exception:
            print(f"Downloading model {model_dir_or_id} from HuggingFace...")
            model_dir = snapshot_download(
                repo_id=model_dir_or_id,
                allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]
            )
            print(f"Downloaded model to {model_dir}")

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Initializing VelocityAI LlamaModel with config: hidden_size={config.get('hidden_size')}, "
          f"num_layers={config.get('num_hidden_layers')}, num_heads={config.get('num_attention_heads')}, "
          f"num_kv_heads={config.get('num_key_value_heads')}...")
    model = LlamaModel(config=config, init_weights=False)

    # Check for optimized pre-transposed cache
    cache_path = os.path.join(model_dir, "optimized.velocityai.safetensors")
    
    if os.path.exists(cache_path):
        print("Found optimized VelocityAI cache. Loading via zero-copy mmap...")
        from velocityai.mmap_loader import mmap_safetensors
        state_dict = mmap_safetensors(cache_path)
    else:
        print("Optimized cache not found. Building it now (this only happens once)...")
        files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors") and not f.endswith("velocityai.safetensors")]
        files.sort()
        
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

        raw_state_dict = {}
        try:
            import mlx.core as mx
            has_mlx = True
        except ImportError:
            has_mlx = False

        for sf in files:
            print(f"Loading raw weights from {os.path.basename(sf)}...")
            if has_mlx:
                tensors_mlx = mx.load(sf)
                for k, v in tensors_mlx.items():
                    raw_state_dict[k] = np.array(v.astype(mx.float32))
            else:
                raw_state_dict.update(load_file(sf))
                
        print("Standardizing and pre-fusing weights in native Float16 precision...")
        state_dict = {}
        
        # 1. Embeddings: (vocab_size, hidden_size) in float16
        if "model.embed_tokens.weight" in raw_state_dict:
            state_dict["model.embed_tokens.weight"] = np.ascontiguousarray(raw_state_dict["model.embed_tokens.weight"]).astype(np.float16)

        # 2. Layers
        for i in range(config.get('num_hidden_layers', 32)):
            prefix = f"model.layers.{i}."
            
            # Norms: (hidden_size) in float32
            if prefix + "input_layernorm.weight" in raw_state_dict:
                state_dict[prefix + "input_layernorm.weight"] = np.ascontiguousarray(raw_state_dict[prefix + "input_layernorm.weight"]).astype(np.float32)
            if prefix + "post_attention_layernorm.weight" in raw_state_dict:
                state_dict[prefix + "post_attention_layernorm.weight"] = np.ascontiguousarray(raw_state_dict[prefix + "post_attention_layernorm.weight"]).astype(np.float32)
                
            # Pre-fuse QKV: (num_heads + 2*num_kv_heads)*head_dim, hidden_size = (1600, 960) in float16
            wq = raw_state_dict[prefix + "self_attn.q_proj.weight"]
            wk = raw_state_dict[prefix + "self_attn.k_proj.weight"]
            wv = raw_state_dict[prefix + "self_attn.v_proj.weight"]
            wqkv = np.ascontiguousarray(np.vstack([wq, wk, wv])).astype(np.float16)
            state_dict[prefix + "self_attn.qkv_proj.weight"] = wqkv
            
            # O proj: (hidden_size, hidden_size) in float16
            wo = np.ascontiguousarray(raw_state_dict[prefix + "self_attn.o_proj.weight"]).astype(np.float16)
            state_dict[prefix + "self_attn.o_proj.weight"] = wo
            
            # Pre-fuse Gate+Up: (2 * intermediate, hidden_size) = (5120, 960) in float16
            w_gate = raw_state_dict[prefix + "mlp.gate_proj.weight"]
            w_up = raw_state_dict[prefix + "mlp.up_proj.weight"]
            w_gate_up = np.ascontiguousarray(np.vstack([w_gate, w_up])).astype(np.float16)
            state_dict[prefix + "mlp.gate_up_proj.weight"] = w_gate_up
            
            # Down proj: (hidden_size, intermediate) = (960, 2560) in float16
            w_down = np.ascontiguousarray(raw_state_dict[prefix + "mlp.down_proj.weight"]).astype(np.float16)
            state_dict[prefix + "mlp.down_proj.weight"] = w_down

        # 3. Final Norm
        if "model.norm.weight" in raw_state_dict:
            state_dict["model.norm.weight"] = np.ascontiguousarray(raw_state_dict["model.norm.weight"]).astype(np.float32)

        # 4. LM Head (if not tied)
        if "lm_head.weight" in raw_state_dict and not config.get("tie_word_embeddings", True):
            state_dict["lm_head.weight"] = np.ascontiguousarray(raw_state_dict["lm_head.weight"]).astype(np.float16)
            
        print(f"Saving optimized FP16 cache to {cache_path}...")
        from safetensors.numpy import save_file
        save_file(state_dict, cache_path)
        print("Loading newly created cache via mmap...")
        from velocityai.mmap_loader import mmap_safetensors
        state_dict = mmap_safetensors(cache_path)

    import time
    t0 = time.time()
    print("Mapping weights to VelocityAI modules...")
    
    # 1. Embeddings
    if "model.embed_tokens.weight" in state_dict:
        model.embed_tokens.weight = vai.from_numpy(state_dict["model.embed_tokens.weight"])
    print(f"Embeddings mapped in {time.time()-t0:.2f}s")
    t1 = time.time()
    
    # 2. Layers
    for i, layer in enumerate(model.layers):
        prefix = f"model.layers.{i}."
        
        if prefix + "input_layernorm.weight" in state_dict:
            layer.input_layernorm.weight = vai.from_numpy(state_dict[prefix + "input_layernorm.weight"])
        if prefix + "post_attention_layernorm.weight" in state_dict:
            layer.post_attention_layernorm.weight = vai.from_numpy(state_dict[prefix + "post_attention_layernorm.weight"])
            
        if prefix + "self_attn.qkv_proj.weight" in state_dict:
            layer.self_attn.qkv_proj = Linear(layer.self_attn.dim, state_dict[prefix + "self_attn.qkv_proj.weight"].shape[0], bias=False, init_weights=False)
            layer.self_attn.qkv_proj.weight = vai.from_numpy(state_dict[prefix + "self_attn.qkv_proj.weight"])
        if prefix + "self_attn.o_proj.weight" in state_dict:
            layer.self_attn.o_proj.weight = vai.from_numpy(state_dict[prefix + "self_attn.o_proj.weight"])
            
        if prefix + "mlp.gate_up_proj.weight" in state_dict:
            layer.mlp.gate_up_proj = Linear(layer.mlp.dim, state_dict[prefix + "mlp.gate_up_proj.weight"].shape[0], bias=False, init_weights=False)
            layer.mlp.gate_up_proj.weight = vai.from_numpy(state_dict[prefix + "mlp.gate_up_proj.weight"])
        if prefix + "mlp.down_proj.weight" in state_dict:
            layer.mlp.down_proj.weight = vai.from_numpy(state_dict[prefix + "mlp.down_proj.weight"])

    print(f"Layers mapped in {time.time()-t1:.2f}s")
    t2 = time.time()
    
    # 3. Final Norm
    if "model.norm.weight" in state_dict:
        model.norm.weight = vai.from_numpy(state_dict["model.norm.weight"])

    # 4. LM Head
    if "lm_head.weight" in state_dict and not model.tie_word_embeddings:
        model.lm_head.weight = vai.from_numpy(state_dict["lm_head.weight"])
        
    print(f"Norm and LM Head mapped in {time.time()-t2:.2f}s")
        
    # Prevent garbage collection of mmap by attaching to model
    if '_mmap' in state_dict:
        model._mmap_cache = state_dict

    print("Model weights successfully loaded into VelocityAI!")
    return model, config
