import time
import subprocess
import json

prompt = "Explain the theory of relativity in simple terms."
model_name = "llama3.2:1b"

def test_ollama():
    print(f"--- Running Ollama ({model_name}) ---")
    
    # Check if model exists
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
        # Pull model in background (ignoring errors if it already exists)
        print(f"Pulling {model_name} (this may take a minute if not downloaded)...")
        subprocess.run(["ollama", "pull", model_name], capture_output=True)
    except FileNotFoundError:
        print("Ollama is not installed or not in PATH.")
        return None
        
    start_time = time.time()
    
    # Run ollama run and capture output
    result = subprocess.run(
        ["ollama", "run", model_name, prompt], 
        capture_output=True, text=True
    )
    
    total_time = time.time() - start_time
    output = result.stdout.strip()
    
    word_count = len(output.split())
    tokens = word_count * 1.3
    tps = tokens / total_time
    
    print(f"Output:\n{output}\n")
    print(f"Time: {total_time:.2f}s")
    print(f"Est. Speed: {tps:.2f} tokens/sec\n")
    
    return {
        "framework": "Ollama",
        "output": output,
        "time": total_time,
        "tps": tps
    }

def test_mlx():
    print("--- Running MLX (mlx-lm) ---")
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("mlx-lm not installed. Installing it now...")
        subprocess.run(["pip", "install", "mlx-lm", "transformers", "huggingface_hub"], check=True)
        from mlx_lm import load, generate

    mlx_model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    
    print(f"Loading {mlx_model_id} via MLX...")
    load_start = time.time()
    model, tokenizer = load(mlx_model_id)
    print(f"Model loaded in {time.time() - load_start:.2f}s")
    
    gen_start = time.time()
    response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=256)
    gen_time = time.time() - gen_start
    
    word_count = len(response.split())
    tokens = word_count * 1.3
    tps = tokens / gen_time
    
    print(f"Output:\n{response}\n")
    print(f"Time (Generation only): {gen_time:.2f}s")
    print(f"Est. Speed: {tps:.2f} tokens/sec\n")
    
    return {
        "framework": "MLX",
        "output": response,
        "time": gen_time,
        "tps": tps
    }

def run_benchmarks():
    print("Starting LLM Inference Benchmarks...\n")
    print(f"Prompt: '{prompt}'\n")
    
    ollama_stats = test_ollama()
    mlx_stats = test_mlx()
    
    print("--- Final Comparison ---")
    if ollama_stats:
        print(f"Ollama Speed: {ollama_stats['tps']:.2f} tok/sec")
    if mlx_stats:
        print(f"MLX Speed:    {mlx_stats['tps']:.2f} tok/sec")
        
    print("\n[Note on VelocityAI]: VelocityAI is a core tensor/autograd compiler (equivalent to PyTorch/MLX core).")
    print("To run Llama 3 natively on VelocityAI, we need to implement the Transformer architecture (Phase 7),")
    print("including RoPE, RMSNorm, KV Cache, and a .safetensors weight loader, which requires massive expansion.")

if __name__ == "__main__":
    run_benchmarks()
