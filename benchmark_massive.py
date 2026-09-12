import os
import sys
import time
import json
import traceback
import urllib.request
import numpy as np

# Suppress HuggingFace logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODELS = [
    {
        "hf_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "ollama_id": "smollm2:360m"
    },
    {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "ollama_id": "qwen2.5:1.5b"
    }
]

PROMPT_COUNT = 100
RESULTS_FILE = "massive_benchmark_results.json"

def load_prompts():
    print("Loading instruction dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        # Filter for open-qa or general brainstorming to get long responses
        filtered = ds.filter(lambda x: x["category"] in ["open_qa", "brainstorming", "creative_writing"])
        prompts = []
        for row in filtered:
            if len(prompts) >= PROMPT_COUNT:
                break
            # Combine instruction and context
            p = row["instruction"]
            if row["context"]:
                p += "\nContext: " + row["context"]
            prompts.append(p)
        return prompts
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Falling back to synthetic prompts for demonstration...")
        return [f"Explain the detailed history and significance of scientific discovery #{i} in at least 3 paragraphs." for i in range(PROMPT_COUNT)]

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def benchmark_ollama(model_name: str, prompt: str):
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": -1
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        raw_response = resp.read().decode("utf-8")
        result = json.loads(raw_response)
    wall_time = time.time() - t0
    
    total_duration = result.get("total_duration", 0) / 1e9
    load_duration = result.get("load_duration", 0) / 1e9
    eval_duration = result.get("eval_duration", 0) / 1e9
    eval_count = result.get("eval_count", 0)
    
    tps = (eval_count / eval_duration) if eval_duration > 0 else 0
    return {
        "framework": "Ollama",
        "load_time_sec": load_duration,
        "decode_time_sec": eval_duration,
        "total_wall_sec": wall_time,
        "generated_tokens": eval_count,
        "tps": tps,
        "output_text": result.get("response", ""),
        "precision": "4-bit (GGUF)"
    }

def benchmark_mlx(model_id: str, prompt: str, model_cache):
    from mlx_lm import load, generate
    if model_id not in model_cache:
        t0 = time.time()
        model_cache[model_id] = load(model_id)
        model_cache[model_id+"_load_time"] = time.time() - t0
        
    model, tokenizer = model_cache[model_id]
    load_time = model_cache[model_id+"_load_time"]
    
    formatted_prompt = prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        
    t_gen_start = time.time()
    # High ceiling, stops on EOS token
    max_tokens = getattr(model.args, 'max_position_embeddings', 8192)
    response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=max_tokens, verbose=False)
    gen_time = time.time() - t_gen_start
    
    output_tokens = len(tokenizer.encode(response))
    tps = output_tokens / gen_time if gen_time > 0 else 0.0
    
    return {
        "framework": "MLX",
        "load_time_sec": load_time,
        "decode_time_sec": gen_time,
        "total_wall_sec": load_time + gen_time,
        "generated_tokens": output_tokens,
        "tps": tps,
        "output_text": response,
        "precision": "fp16"
    }

def benchmark_pytorch(model_id: str, prompt: str, model_cache):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if model_id not in model_cache:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cpu")
        model_cache[model_id] = (model, tokenizer)
        model_cache[model_id+"_load_time"] = time.time() - t0
        
    model, tokenizer = model_cache[model_id]
    load_time = model_cache[model_id+"_load_time"]
    
    formatted_prompt = prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    t_gen_start = time.time()
    with torch.no_grad():
        max_tokens = getattr(model.config, 'max_position_embeddings', 8192)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
    gen_time = time.time() - t_gen_start
    
    output_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    tps = output_tokens / gen_time if gen_time > 0 else 0.0
    
    return {
        "framework": "PyTorch",
        "load_time_sec": load_time,
        "decode_time_sec": gen_time,
        "total_wall_sec": load_time + gen_time,
        "generated_tokens": output_tokens,
        "tps": tps,
        "output_text": output_text,
        "precision": "fp16"
    }

def _velocityai_worker(model_id, prompt, return_dict):
    import time
    import velocityai as vai
    import sys
    import os
    
    # Suppress stdout to avoid spamming terminal on every prompt
    sys.stdout = open(os.devnull, 'w')
    
    t0 = time.time()
    model, config = vai.load_huggingface_model(model_id)
    tokenizer = vai.Tokenizer(model_id)
    
    # WARMUP: Apple Accelerate's GCD thread pool takes ~2.5s to initialize on the very first 
    # cblas_sgemm call. Do a 1-token dummy pass so this initialization overhead 
    # correctly counts towards load_time, not prefill latency.
    dummy = vai.tensor(np.array([[0]], dtype=np.int64)).to("cpu")
    model(dummy, start_pos=0, last_token_only=True)
    
    load_time = time.time() - t0
    
    t_gen_start = time.time()
    max_tokens = getattr(config, 'max_position_embeddings', 8192)
    res = vai.generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.0,
        device="cpu"
    )
    wall_time_for_generate = time.time() - t_gen_start
    warmup_time = wall_time_for_generate - res.get("total_time", 0.0)
    
    return_dict["result"] = {
        "framework": "VelocityAI",
        "load_time_sec": load_time,
        "warmup_time_sec": warmup_time,
        "prefill_time_sec": res.get("ttft", 0.0),
        "decode_time_sec": res.get("decode_time", res.get("total_time", 0.0)),
        "total_wall_sec": load_time + wall_time_for_generate,
        "generated_tokens": res["output_tokens_count"],
        "tps": res.get("decode_tps", res["tps"]),
        "output_text": res.get("output_text", res.get("text", res.get("response", ""))),
        "precision": "fp16"
    }

def benchmark_velocityai(model_id: str, prompt: str, model_cache):
    import multiprocessing
    from transformers import AutoTokenizer
    
    if model_id not in model_cache:
        model_cache[model_id] = AutoTokenizer.from_pretrained(model_id)
    
    hf_tokenizer = model_cache[model_id]
    formatted_prompt = prompt
    if hasattr(hf_tokenizer, "apply_chat_template") and hf_tokenizer.chat_template:
        formatted_prompt = hf_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    p = multiprocessing.Process(target=_velocityai_worker, args=(model_id, formatted_prompt, return_dict))
    p.start()
    p.join()
    
    if p.exitcode != 0:
        raise RuntimeError(f"VelocityAI C-extension crashed (exit code {p.exitcode}, likely SIGABRT)")
        
    if "result" in return_dict:
        return return_dict["result"]
    else:
        raise RuntimeError("VelocityAI process did not return a result")

def main():
    print(f"Starting MASSIVE 1000-Prompt 4-Model LLM Benchmark")
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts.")
    
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    else:
        results = {m["hf_id"]: {"VelocityAI": [], "MLX": [], "PyTorch": [], "Ollama": []} for m in MODELS}
        
    model_caches = {"MLX": {}, "PyTorch": {}, "VelocityAI": {}}
    
    try:
        for model_info in MODELS:
            hf_id = model_info["hf_id"]
            ollama_id = model_info["ollama_id"]
            print(f"\n{'='*50}\nEvaluating Model: {hf_id}\n{'='*50}")
            
            for i, prompt in enumerate(prompts):
                print(f"  [Prompt {i+1}/{len(prompts)}] Model: {hf_id}")
                
                # 1. VelocityAI
                if len(results[hf_id]["VelocityAI"]) <= i:
                    try:
                        res = benchmark_velocityai(hf_id, prompt, model_caches["VelocityAI"])
                        results[hf_id]["VelocityAI"].append(res)
                        print(f"    - VelocityAI: {res['tps']:.2f} tok/s")
                    except Exception as e:
                        print(f"    - VelocityAI FAILED: {e}")
                        results[hf_id]["VelocityAI"].append({"error": str(e)})
                
                # 2. MLX
                if len(results[hf_id]["MLX"]) <= i:
                    try:
                        res = benchmark_mlx(hf_id, prompt, model_caches["MLX"])
                        results[hf_id]["MLX"].append(res)
                        print(f"    - MLX: {res['tps']:.2f} tok/s")
                    except Exception as e:
                        print(f"    - MLX FAILED: {e}")
                        results[hf_id]["MLX"].append({"error": str(e)})
                        
                # 3. PyTorch
                if len(results[hf_id]["PyTorch"]) <= i:
                    try:
                        res = benchmark_pytorch(hf_id, prompt, model_caches["PyTorch"])
                        results[hf_id]["PyTorch"].append(res)
                        print(f"    - PyTorch: {res['tps']:.2f} tok/s")
                    except Exception as e:
                        print(f"    - PyTorch FAILED: {e}")
                        results[hf_id]["PyTorch"].append({"error": str(e)})
                        
                # 4. Ollama
                if len(results[hf_id]["Ollama"]) <= i:
                    try:
                        res = benchmark_ollama(ollama_id, prompt)
                        results[hf_id]["Ollama"].append(res)
                        print(f"    - Ollama: {res['tps']:.2f} tok/s")
                    except Exception as e:
                        print(f"    - Ollama FAILED: {e}")
                        results[hf_id]["Ollama"].append({"error": str(e)})
                
                # Save checkpoint every 10 prompts
                if (i + 1) % 10 == 0:
                    save_results(results)
                    print(f"  --> Checkpoint Saved ({i+1}/{len(prompts)})")
                    
            save_results(results)
            # Clear caches to avoid Out-Of-Memory when switching models
            model_caches = {"MLX": {}, "PyTorch": {}, "VelocityAI": {}}
            import gc
            gc.collect()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user. Progress saved.")
    except Exception as e:
        traceback.print_exc()
        print(f"\nCritical failure: {e}")
    finally:
        save_results(results)
        print(f"\nBenchmark finished/paused. Results stored in {RESULTS_FILE}")

if __name__ == "__main__":
    main()
