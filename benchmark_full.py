import os
import sys
import time
import json
import urllib.request
import urllib.parse
import numpy as np

# Prompt for the benchmark
PROMPT = "Explain the difference between gradient descent and stochastic gradient descent."
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0

def benchmark_ollama(model_name: str = "smollm2:360m", prompt: str = PROMPT, max_tokens: int = MAX_NEW_TOKENS):
    print(f"\n{'='*20} 1. BENCHMARKING OLLAMA ({model_name}) {'='*20}")
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": TEMPERATURE,
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            raw_response = resp.read().decode("utf-8")
            result = json.loads(raw_response)
    except Exception as e:
        print(f"Ollama request failed: {e}")
        return None
        
    wall_time = time.time() - t0
    
    # Extract timings from Ollama response (nanoseconds to seconds)
    total_duration = result.get("total_duration", 0) / 1e9
    load_duration = result.get("load_duration", 0) / 1e9
    prompt_eval_duration = result.get("prompt_eval_duration", 0) / 1e9
    eval_duration = result.get("eval_duration", 0) / 1e9
    eval_count = result.get("eval_count", 0)
    prompt_eval_count = result.get("prompt_eval_count", 0)
    
    tps = (eval_count / eval_duration) if eval_duration > 0 else (eval_count / wall_time)
    output_text = result.get("response", "").strip()
    
    stats = {
        "framework": "Ollama",
        "model": model_name,
        "format": "GGUF Q4_K_M",
        "load_time_sec": round(load_duration, 4),
        "prompt_eval_sec": round(prompt_eval_duration, 4),
        "eval_time_sec": round(eval_duration, 4),
        "total_wall_sec": round(wall_time, 4),
        "prompt_tokens": prompt_eval_count,
        "generated_tokens": eval_count,
        "tps": round(tps, 2),
        "output_text": output_text
    }
    
    print(f"Tokens Generated: {eval_count}")
    print(f"Generation Speed: {tps:.2f} tokens/sec")
    print(f"Load Duration:    {load_duration:.4f}s")
    print(f"Eval Duration:    {eval_duration:.4f}s")
    print(f"Output:\n{output_text}\n")
    return stats

def benchmark_mlx(model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct", prompt: str = PROMPT, max_tokens: int = MAX_NEW_TOKENS):
    print(f"\n{'='*20} 2. BENCHMARKING MLX ({model_id}) {'='*20}")
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("mlx-lm not installed!")
        return None
        
    t_load_start = time.time()
    print(f"Loading MLX model: {model_id}...")
    model, tokenizer = load(model_id)
    load_time = time.time() - t_load_start
    print(f"MLX Model Loaded in {load_time:.2f}s")
    
    # Warmup
    _ = generate(model, tokenizer, prompt="Hi", max_tokens=2, verbose=False)
    
    # Generate
    t_gen_start = time.time()
    formatted_prompt = prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        
    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        verbose=False
    )
    gen_time = time.time() - t_gen_start
    
    # Count generated tokens
    output_tokens = len(tokenizer.encode(response))
    tps = output_tokens / gen_time if gen_time > 0 else 0.0
    
    stats = {
        "framework": "MLX (Apple Silicon)",
        "model": model_id,
        "format": "BFloat16 (Metal)",
        "load_time_sec": round(load_time, 4),
        "eval_time_sec": round(gen_time, 4),
        "total_wall_sec": round(load_time + gen_time, 4),
        "generated_tokens": output_tokens,
        "tps": round(tps, 2),
        "output_text": response.strip()
    }
    
    print(f"Tokens Generated: {output_tokens}")
    print(f"Generation Speed: {tps:.2f} tokens/sec")
    print(f"Eval Duration:    {gen_time:.4f}s")
    print(f"Output:\n{response.strip()}\n")
    return stats

def benchmark_velocityai(model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct", prompt: str = PROMPT, max_tokens: int = MAX_NEW_TOKENS):
    print(f"\n{'='*20} 3. BENCHMARKING VelocityAI ({model_id}) {'='*20}")
    import velocityai as vai
    
    t_load_start = time.time()
    print(f"Loading VelocityAI model: {model_id}...")
    model, config = vai.load_huggingface_model(model_id)
    tokenizer = vai.Tokenizer(model_id)
    load_time = time.time() - t_load_start
    print(f"VelocityAI Model Loaded in {load_time:.2f}s")
    
    # Run generation
    print("Generating tokens with VelocityAI autoregressive decoder (Apple Accelerate AMX)...")
    res = vai.generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=TEMPERATURE,
        device="cpu"
    )
    
    stats = {
        "framework": "VelocityAI (Ours)",
        "model": model_id,
        "format": "FP16 SafeTensors (Fused C++ NEON)",
        "load_time_sec": round(load_time, 4),
        "ttft_sec": round(res.get("ttft", 0), 4),
        "eval_time_sec": round(res["total_time"], 4),
        "decode_time_sec": round(res.get("decode_time", 0), 4),
        "total_wall_sec": round(load_time + res["total_time"], 4),
        "prompt_tokens": res["input_tokens_count"],
        "generated_tokens": res["output_tokens_count"],
        "tps": round(res["tps"], 2),
        "decode_tps": round(res.get("decode_tps", res["tps"]), 2),
        "output_text": res["text"].strip()
    }
    
    print(f"Tokens Generated: {res['output_tokens_count']}")
    print(f"TTFT (Prefill):   {res.get('ttft', 0):.4f}s")
    print(f"Decode Duration:  {res.get('decode_time', 0):.4f}s")
    print(f"Decode Speed:     {res.get('decode_tps', 0):.2f} tokens/sec")
    print(f"Overall Speed:    {res['tps']:.2f} tokens/sec")
    print(f"Output:\n{res['text'].strip()}\n")
    return stats

def main():
    print(f"Starting 3-Way LLM Benchmark across Ollama, MLX, and VelocityAI")
    print(f"Prompt: '{PROMPT}'")
    print(f"Target Max Tokens: {MAX_NEW_TOKENS}")
    
    results = {}
    
    # 1. Ollama
    try:
        results["ollama"] = benchmark_ollama()
    except Exception as e:
        print(f"Ollama failed: {e}")
        
    # 2. MLX
    try:
        results["mlx"] = benchmark_mlx()
    except Exception as e:
        print(f"MLX failed: {e}")
        
    # 3. VelocityAI
    try:
        results["velocityai"] = benchmark_velocityai()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"VelocityAI failed: {e}")
        
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Print summary table
    print("\n" + "="*95)
    print(f"{'FRAMEWORK':<20} | {'FORMAT':<28} | {'LOAD (s)':<10} | {'DECODE (s)':<12} | {'DECODE TOK/S':<14}")
    print("-"*95)
    for k, v in results.items():
        if v:
            tps_val = v.get('decode_tps', v.get('tps', 0))
            dec_time = v.get('decode_time_sec', v.get('eval_time_sec', 0))
            print(f"{v['framework']:<20} | {v['format']:<28} | {v['load_time_sec']:<10.2f} | {dec_time:<12.4f} | {tps_val:<14.2f}")
    print("="*95)

if __name__ == "__main__":
    main()
