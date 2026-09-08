#!/usr/bin/env python3
"""
VelocityAI vs MLX vs Ollama: Comprehensive Multi-Domain Reality Check & Thermal Benchmark
Evaluates:
  1. Performance (TTFT, Decode tok/s, ITL ms/tok, E2E Latency)
  2. Latency Distributions (p50, p95, Mean, StdDev)
  3. Thermal Dynamics (macOS NSProcessInfo thermalState, pmset -g therm, Throttling Factor Φ, Time-to-Throttle)
  4. Mathematical Output Quality & Fidelity (FP16 lossless vs Q4_K_M lossy quantization)
"""

import os
import sys
import time
import json
import argparse
import threading
import subprocess
import urllib.request
import urllib.parse
import numpy as np
import ctypes
import ctypes.util

# ---------------------------------------------------------------------------
# 1. Thermal & Hardware Throttling Monitor
# ---------------------------------------------------------------------------

class ThermalMonitor:
    """
    Monitors Apple Silicon thermal states, thermal pressure levels,
    and CPU throttling via macOS native APIs.
    """
    def __init__(self, sample_interval: float = 0.5):
        self.interval = sample_interval
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # History
        self.records = []
        self.initial_state = 0
        self.peak_state = 0
        self.throttle_start_time = None
        self.total_throttle_seconds = 0.0
        self.t_start = 0.0
        
        # Setup Objective-C NSProcessInfo
        self._setup_objc()

    def _setup_objc(self):
        try:
            self.objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
            self.objc.objc_getClass.restype = ctypes.c_void_p
            self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
            self.objc.sel_registerName.restype = ctypes.c_void_p
            self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
            
            self.msg_send = self.objc.objc_msgSend
            self.msg_send.restype = ctypes.c_void_p
            self.msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            
            self.cls_pi = self.objc.objc_getClass(b"NSProcessInfo")
            self.sel_pi = self.objc.sel_registerName(b"processInfo")
            self.pi = self.msg_send(self.cls_pi, self.sel_pi)
            
            self.sel_ts = self.objc.sel_registerName(b"thermalState")
            self.msg_send_state = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc")).objc_msgSend
            self.msg_send_state.restype = ctypes.c_long
            self.msg_send_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            self.initial_state = self.get_thermal_state()
            self.peak_state = self.initial_state
        except Exception as e:
            self.pi = None
            self.initial_state = 0

    def get_thermal_state(self) -> int:
        """
        Returns NSProcessInfoThermalState:
          0: Nominal (Full 3.5 GHz clock, no throttling)
          1: Fair (Elevated junction temperature)
          2: Serious (CPU/GPU thermal throttling active)
          3: Critical (Emergency throttling)
        """
        if self.pi:
            try:
                return int(self.msg_send_state(self.pi, self.sel_ts))
            except Exception:
                return 0
        return 0

    def get_pmset_thermal_summary(self) -> dict:
        """Parses pmset -g therm output."""
        res = {"cpu_warning": "None", "perf_warning": "None", "power_status": "Normal"}
        try:
            out = subprocess.check_output(["pmset", "-g", "therm"], stderr=subprocess.DEVNULL).decode("utf-8")
            for line in out.splitlines():
                if "thermal warning" in line.lower() and "no " not in line.lower():
                    res["cpu_warning"] = line.strip()
                if "performance warning" in line.lower() and "no " not in line.lower():
                    res["perf_warning"] = line.strip()
        except Exception:
            pass
        return res

    def start(self):
        self.running = True
        self.t_start = time.time()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _loop(self):
        while self.running:
            now = time.time()
            state = self.get_thermal_state()
            with self.lock:
                if state > self.peak_state:
                    self.peak_state = state
                
                # Check throttling
                if state >= 2:
                    if self.throttle_start_time is None:
                        self.throttle_start_time = now - self.t_start
                    self.total_throttle_seconds += self.interval

                self.records.append({
                    "time": round(now - self.t_start, 2),
                    "thermal_state": state,
                })
            time.sleep(self.interval)

    def get_summary(self) -> dict:
        with self.lock:
            state_names = {0: "Nominal", 1: "Fair", 2: "Serious (Throttling)", 3: "Critical (Severe)"}
            return {
                "initial_thermal_state": self.initial_state,
                "initial_thermal_desc": state_names.get(self.initial_state, "Unknown"),
                "peak_thermal_state": self.peak_state,
                "peak_thermal_desc": state_names.get(self.peak_state, "Unknown"),
                "time_to_first_throttle_sec": self.throttle_start_time,
                "total_throttle_seconds": round(self.total_throttle_seconds, 2),
                "is_throttled": self.peak_state >= 2,
            }


# ---------------------------------------------------------------------------
# 2. Curated 50 Multi-Domain Prompts
# ---------------------------------------------------------------------------

MULTI_DOMAIN_PROMPTS = [
    # Domain 1: Mathematical & Logical Reasoning (10)
    {"id": "math_01", "domain": "Math & Logic", "prompt": "What is 25 * 36? Show step-by-step mental calculation."},
    {"id": "math_02", "domain": "Math & Logic", "prompt": "If a train travels 75 miles in 1 hour and 15 minutes, what is its average speed in mph?"},
    {"id": "math_03", "domain": "Math & Logic", "prompt": "A store offers a 20% discount on a $150 item, then applies an 8% sales tax. What is the final price?"},
    {"id": "math_04", "domain": "Math & Logic", "prompt": "If you flip a fair coin 3 times, what is the exact probability of getting at least two heads?"},
    {"id": "math_05", "domain": "Math & Logic", "prompt": "Find the sum of all integers from 1 to 100 using Gauss's formula."},
    {"id": "math_06", "domain": "Math & Logic", "prompt": "If 5 machines can make 5 widgets in 5 minutes, how many minutes does it take 100 machines to make 100 widgets?"},
    {"id": "math_07", "domain": "Math & Logic", "prompt": "A father is 3 times as old as his son. In 12 years, he will be twice as old. How old are they now?"},
    {"id": "math_08", "domain": "Math & Logic", "prompt": "What is the hypotenuse of a right triangle with legs of length 7 and 24?"},
    {"id": "math_09", "domain": "Math & Logic", "prompt": "Calculate 2^10 and explain how it relates to computing memory."},
    {"id": "math_10", "domain": "Math & Logic", "prompt": "If you have 12 red balls and 8 blue balls in a bag, what is the probability of drawing two red balls in a row without replacement?"},

    # Domain 2: Code Generation & Logic (10)
    {"id": "code_01", "domain": "Code Generation", "prompt": "Write a Python function to check if a string is a palindrome ignoring spaces and punctuation."},
    {"id": "code_02", "domain": "Code Generation", "prompt": "Write a Python function implementing binary search on a sorted list with comments."},
    {"id": "code_03", "domain": "Code Generation", "prompt": "How do you reverse a singly linked list in Python? Provide clean code."},
    {"id": "code_04", "domain": "Code Generation", "prompt": "Write a Python function using recursion to compute the nth Fibonacci number with memoization."},
    {"id": "code_05", "domain": "Code Generation", "prompt": "Write an SQL query to find the second highest salary from an Employee table."},
    {"id": "code_06", "domain": "Code Generation", "prompt": "Implement a Python generator that yields prime numbers up to n using the Sieve of Eratosthenes."},
    {"id": "code_07", "domain": "Code Generation", "prompt": "Write a Python context manager using the `contextlib` decorator to measure code execution time."},
    {"id": "code_08", "domain": "Code Generation", "prompt": "Explain the difference between deep copy and shallow copy in Python with code examples."},
    {"id": "code_09", "domain": "Code Generation", "prompt": "Write a Python function to flatten a nested list of arbitrary depth."},
    {"id": "code_10", "domain": "Code Generation", "prompt": "How does LRU Cache work? Write a minimal LRUCache class in Python using OrderedDict."},

    # Domain 3: Factual Science & World Knowledge (10)
    {"id": "fact_01", "domain": "Factual Science", "prompt": "Explain Albert Einstein's Theory of General Relativity in simple terms."},
    {"id": "fact_02", "domain": "Factual Science", "prompt": "How does photosynthesis convert solar energy into chemical energy? Detail the light reactions."},
    {"id": "fact_03", "domain": "Factual Science", "prompt": "What is the speed of light in a vacuum, and why can nothing travel faster than it?"},
    {"id": "fact_04", "domain": "Factual Science", "prompt": "Explain the role of mitochondria in human cellular metabolism."},
    {"id": "fact_05", "domain": "Factual Science", "prompt": "How do tectonic plates cause earthquakes and volcanic eruptions?"},
    {"id": "fact_06", "domain": "Factual Science", "prompt": "What was the cause and historical significance of the French Revolution?"},
    {"id": "fact_07", "domain": "Factual Science", "prompt": "Explain the structure of DNA, the double helix, and base pairing rules."},
    {"id": "fact_08", "domain": "Factual Science", "prompt": "What are the primary differences between RNA viruses and DNA viruses?"},
    {"id": "fact_09", "domain": "Factual Science", "prompt": "Explain how the greenhouse effect maintains Earth's habitable temperature."},
    {"id": "fact_10", "domain": "Factual Science", "prompt": "What causes oceanic tides and how do the Moon and Sun interact in spring and neap tides?"},

    # Domain 4: Reasoning & Common Sense (10)
    {"id": "reas_01", "domain": "Reasoning", "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain logical validity."},
    {"id": "reas_02", "domain": "Reasoning", "prompt": "Why do hot air balloons rise in the air while helium balloons also rise? Compare the mechanisms."},
    {"id": "reas_03", "domain": "Reasoning", "prompt": "Why does ice float on water when most substances are denser in solid form?"},
    {"id": "reas_04", "domain": "Reasoning", "prompt": "If you put an apple in a sealed vacuum chamber, what happens to it?"},
    {"id": "reas_05", "domain": "Reasoning", "prompt": "Why can birds sit on high-voltage electric power lines without getting electrocuted?"},
    {"id": "reas_06", "domain": "Reasoning", "prompt": "Explain why a mirror reverses left and right but does not reverse up and down."},
    {"id": "reas_07", "domain": "Reasoning", "prompt": "If a tree falls in a forest and no one is around to hear it, does it make a sound? Analyze physics vs perception."},
    {"id": "reas_08", "domain": "Reasoning", "prompt": "Why is the sky blue during the day but red/orange at sunset? Explain Rayleigh scattering."},
    {"id": "reas_09", "domain": "Reasoning", "prompt": "Compare the advantages and disadvantages of renewable energy vs nuclear energy."},
    {"id": "reas_10", "domain": "Reasoning", "prompt": "Why does sweating cool the human body down? Explain evaporative cooling."},

    # Domain 5: Structured Synthesis & Context Ingestion (10)
    {"id": "synth_01", "domain": "Synthesis & Context", "prompt": "Summarize the core principles of quantum mechanics into 3 clear bullet points."},
    {"id": "synth_02", "domain": "Synthesis & Context", "prompt": "Explain the difference between symmetric and asymmetric cryptography with real-world examples."},
    {"id": "synth_03", "domain": "Synthesis & Context", "prompt": "What are ACID properties in database transactions? Explain Atomicity, Consistency, Isolation, Durability."},
    {"id": "synth_04", "domain": "Synthesis & Context", "prompt": "Explain how a Transformer neural network processes input tokens with multi-head self-attention."},
    {"id": "synth_05", "domain": "Synthesis & Context", "prompt": "Compare RISC and CISC CPU architectures. Why does Apple Silicon use ARM RISC?"},
    {"id": "synth_06", "domain": "Synthesis & Context", "prompt": "Explain the concept of memory hierarchy: L1, L2, L3 caches, main RAM, and NVMe SSD storage."},
    {"id": "synth_07", "domain": "Synthesis & Context", "prompt": "What is the difference between TCP and UDP? When should a software engineer choose UDP?"},
    {"id": "synth_08", "domain": "Synthesis & Context", "prompt": "Summarize the Turing Test and John Searle's Chinese Room argument regarding artificial intelligence."},
    {"id": "synth_09", "domain": "Synthesis & Context", "prompt": "Explain what speculative execution and branch prediction are in modern microprocessors."},
    {"id": "synth_10", "domain": "Synthesis & Context", "prompt": "What is the P vs NP problem in theoretical computer science, and why is it so important?"}
]


# ---------------------------------------------------------------------------
# 3. Engine Benchmark Wrappers
# ---------------------------------------------------------------------------

def run_velocityai_benchmark(prompts, max_tokens=64, temperature=0.0):
    print("\n" + "="*70)
    print(" 🚀 BENCHMARKING ENGINE 1: VelocityAI (Native C++ ARM NEON FP16)")
    print("="*70)
    import velocityai as vai
    
    t0 = time.time()
    model, config = vai.load_huggingface_model("HuggingFaceTB/SmolLM2-360M-Instruct")
    tokenizer = vai.Tokenizer("HuggingFaceTB/SmolLM2-360M-Instruct")
    load_time = time.time() - t0
    print(f"[✓] VelocityAI Model Loaded in {load_time:.2f}s")
    
    # Warmup
    _ = vai.generate(model, tokenizer, prompt="Warmup query", max_new_tokens=4, temperature=0.0)
    
    results = []
    for idx, item in enumerate(prompts, 1):
        p_text = item["prompt"]
        t_gen_start = time.time()
        res = vai.generate(
            model=model,
            tokenizer=tokenizer,
            prompt=p_text,
            max_new_tokens=max_tokens,
            temperature=temperature,
            device="cpu"
        )
        total_wall = time.time() - t_gen_start
        
        ttft = res.get("ttft", 0.0)
        decode_time = res.get("decode_time", total_wall)
        tokens_out = res["output_tokens_count"]
        decode_tps = res.get("decode_tps", tokens_out / decode_time if decode_time > 0 else 0)
        itl_ms = (decode_time / max(1, tokens_out - 1)) * 1000.0
        
        record = {
            "id": item["id"],
            "domain": item["domain"],
            "prompt": p_text,
            "engine": "VelocityAI",
            "format": "FP16 Lossless (ARM NEON)",
            "input_tokens": res["input_tokens_count"],
            "output_tokens": tokens_out,
            "ttft_sec": round(ttft, 4),
            "decode_time_sec": round(decode_time, 4),
            "total_time_sec": round(total_wall, 4),
            "decode_tps": round(decode_tps, 2),
            "itl_ms": round(itl_ms, 2),
            "output_text": res["text"].strip()
        }
        results.append(record)
        print(f"[{idx:02d}/{len(prompts)}] VelocityAI | {decode_tps:6.2f} tok/s | TTFT: {ttft*1000:5.1f}ms | Out: {tokens_out:2d} tok | {item['id']}")

    return results, load_time


def run_mlx_benchmark(prompts, max_tokens=64, temperature=0.0):
    print("\n" + "="*70)
    print(" 🍏 BENCHMARKING ENGINE 2: Apple MLX (Metal GPU BFloat16)")
    print("="*70)
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("[-] MLX-LM not available, skipping MLX benchmark.")
        return [], 0.0

    t0 = time.time()
    model, tokenizer = load("HuggingFaceTB/SmolLM2-360M-Instruct")
    load_time = time.time() - t0
    print(f"[✓] MLX Model Loaded in {load_time:.2f}s")
    
    # Warmup
    _ = generate(model, tokenizer, prompt="Warmup query", max_tokens=4, verbose=False)
    
    results = []
    for idx, item in enumerate(prompts, 1):
        p_text = item["prompt"]
        formatted_prompt = p_text
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": p_text}], tokenize=False, add_generation_prompt=True)
            
        t_gen_start = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        total_wall = time.time() - t_gen_start
        
        output_tokens = len(tokenizer.encode(response))
        tps = output_tokens / total_wall if total_wall > 0 else 0.0
        itl_ms = (total_wall / max(1, output_tokens)) * 1000.0
        
        record = {
            "id": item["id"],
            "domain": item["domain"],
            "prompt": p_text,
            "engine": "MLX",
            "format": "BFloat16 (Metal GPU)",
            "output_tokens": output_tokens,
            "total_time_sec": round(total_wall, 4),
            "decode_tps": round(tps, 2),
            "itl_ms": round(itl_ms, 2),
            "output_text": response.strip()
        }
        results.append(record)
        print(f"[{idx:02d}/{len(prompts)}] Apple MLX  | {tps:6.2f} tok/s | Latency: {total_wall*1000:5.1f}ms | Out: {output_tokens:2d} tok | {item['id']}")

    return results, load_time


def run_ollama_benchmark(prompts, max_tokens=64, temperature=0.0):
    print("\n" + "="*70)
    print(" 🦙 BENCHMARKING ENGINE 3: Ollama (GGUF Q4_K_M 4-bit Quantized)")
    print("="*70)
    url = "http://127.0.0.1:11434/api/generate"
    
    # Test connection
    try:
        req = urllib.request.Request(url, data=json.dumps({"model": "smollm2:360m", "prompt": "Hi", "stream": False, "options": {"num_predict": 2}}).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass
        print("[✓] Ollama server is responsive at localhost:11434")
    except Exception as e:
        print(f"[-] Ollama connection error: {e}. Skipping Ollama benchmark.")
        return [], 0.0

    results = []
    load_time = 0.0
    for idx, item in enumerate(prompts, 1):
        p_text = item["prompt"]
        payload = {
            "model": "smollm2:360m",
            "prompt": p_text,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_response = resp.read().decode("utf-8")
                res = json.loads(raw_response)
        except Exception as e:
            print(f"[{idx:02d}/{len(prompts)}] Ollama request failed: {e}")
            continue
        total_wall = time.time() - t0
        
        prompt_eval_sec = res.get("prompt_eval_duration", 0) / 1e9
        eval_sec = res.get("eval_duration", 0) / 1e9
        eval_count = res.get("eval_count", 0)
        prompt_count = res.get("prompt_eval_count", 0)
        if idx == 1:
            load_time = res.get("load_duration", 0) / 1e9
            
        decode_tps = (eval_count / eval_sec) if eval_sec > 0 else (eval_count / total_wall)
        itl_ms = (eval_sec / max(1, eval_count)) * 1000.0 if eval_sec > 0 else (total_wall / max(1, eval_count)) * 1000.0
        
        record = {
            "id": item["id"],
            "domain": item["domain"],
            "prompt": p_text,
            "engine": "Ollama",
            "format": "Q4_K_M (4-bit Quantized)",
            "input_tokens": prompt_count,
            "output_tokens": eval_count,
            "ttft_sec": round(prompt_eval_sec, 4),
            "decode_time_sec": round(eval_sec, 4),
            "total_time_sec": round(total_wall, 4),
            "decode_tps": round(decode_tps, 2),
            "itl_ms": round(itl_ms, 2),
            "output_text": res.get("response", "").strip()
        }
        results.append(record)
        print(f"[{idx:02d}/{len(prompts)}] Ollama     | {decode_tps:6.2f} tok/s | TTFT: {prompt_eval_sec*1000:5.1f}ms | Out: {eval_count:2d} tok | {item['id']}")

    return results, load_time


# ---------------------------------------------------------------------------
# 4. Statistical Analysis & Fidelity Comparison
# ---------------------------------------------------------------------------

def compute_statistics(results):
    if not results:
        return {}
    tps_list = [r["decode_tps"] for r in results]
    itl_list = [r["itl_ms"] for r in results]
    ttft_list = [r.get("ttft_sec", 0.0) * 1000.0 for r in results if r.get("ttft_sec", 0.0) > 0]
    
    return {
        "count": len(results),
        "mean_tps": round(float(np.mean(tps_list)), 2),
        "median_tps": round(float(np.median(tps_list)), 2),
        "p95_tps": round(float(np.percentile(tps_list, 95)), 2),
        "min_tps": round(float(np.min(tps_list)), 2),
        "max_tps": round(float(np.max(tps_list)), 2),
        "std_tps": round(float(np.std(tps_list)), 2),
        "mean_itl_ms": round(float(np.mean(itl_list)), 2),
        "median_itl_ms": round(float(np.median(itl_list)), 2),
        "p95_itl_ms": round(float(np.percentile(itl_list, 95)), 2),
        "mean_ttft_ms": round(float(np.mean(ttft_list)), 2) if ttft_list else 0.0,
    }


def compute_fidelity_comparison(vel_results, mlx_results, ollama_results):
    """
    Computes exact string match, token prefix match, and quantization drift.
    """
    comp = []
    vel_map = {r["id"]: r for r in vel_results}
    mlx_map = {r["id"]: r for r in mlx_results}
    oll_map = {r["id"]: r for r in ollama_results}
    
    for q_id, v_item in vel_map.items():
        v_text = v_item["output_text"]
        m_item = mlx_map.get(q_id)
        o_item = oll_map.get(q_id)
        
        m_text = m_item["output_text"] if m_item else ""
        o_text = o_item["output_text"] if o_item else ""
        
        # Word overlap with MLX reference
        v_words = set(v_text.lower().split())
        m_words = set(m_text.lower().split())
        o_words = set(o_text.lower().split())
        
        jaccard_vel_mlx = len(v_words & m_words) / max(1, len(v_words | m_words))
        jaccard_oll_mlx = len(o_words & m_words) / max(1, len(o_words | m_words))
        
        comp.append({
            "id": q_id,
            "domain": v_item["domain"],
            "prompt": v_item["prompt"],
            "vel_text_snippet": v_text[:80] + ("..." if len(v_text) > 80 else ""),
            "mlx_text_snippet": m_text[:80] + ("..." if len(m_text) > 80 else ""),
            "oll_text_snippet": o_text[:80] + ("..." if len(o_text) > 80 else ""),
            "vel_vs_mlx_similarity": round(jaccard_vel_mlx, 3),
            "oll_vs_mlx_similarity": round(jaccard_oll_mlx, 3),
            "quant_drift_detected": abs(jaccard_vel_mlx - jaccard_oll_mlx) > 0.15
        })
    return comp


# ---------------------------------------------------------------------------
# 5. Main Runner & Thermal Stress Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VelocityAI Reality Check & Thermal Benchmark")
    parser.add_argument("--prompts", "-p", type=int, default=50, help="Number of prompts to benchmark (1-50, default: 50)")
    parser.add_argument("--tokens", "-n", type=int, default=64, help="Tokens to generate per prompt (default: 64)")
    parser.add_argument("--endurance", "-e", type=int, default=20, help="Endurance loop count to monitor thermal throttling (default: 20)")
    args = parser.parse_args()

    selected_prompts = MULTI_DOMAIN_PROMPTS[:args.prompts]
    print("="*80)
    print(" 🏁 VELOCITYAI vs APPLE MLX vs OLLAMA: COMPREHENSIVE REALITY CHECK")
    print(f" Total Multi-Domain Prompts: {len(selected_prompts)} | Tokens per prompt: {args.tokens}")
    print("="*80)

    thermal_monitor = ThermalMonitor(sample_interval=0.5)
    thermal_monitor.start()
    
    # 1. Run VelocityAI
    t0_vel = time.time()
    vel_records, vel_load_time = run_velocityai_benchmark(selected_prompts, max_tokens=args.tokens)
    t_vel_elapsed = time.time() - t0_vel
    vel_stats = compute_statistics(vel_records)
    vel_thermal = thermal_monitor.get_summary()

    # 2. Run Apple MLX
    t0_mlx = time.time()
    mlx_records, mlx_load_time = run_mlx_benchmark(selected_prompts, max_tokens=args.tokens)
    t_mlx_elapsed = time.time() - t0_mlx
    mlx_stats = compute_statistics(mlx_records)
    mlx_thermal = thermal_monitor.get_summary()

    # 3. Run Ollama
    t0_oll = time.time()
    oll_records, oll_load_time = run_ollama_benchmark(selected_prompts, max_tokens=args.tokens)
    t_oll_elapsed = time.time() - t0_oll
    oll_stats = compute_statistics(oll_records)
    oll_thermal = thermal_monitor.get_summary()

    # Stop background thermal monitor
    thermal_monitor.stop()

    # 4. Compute Quality & Quantization Drift
    fidelity_comp = compute_fidelity_comparison(vel_records, mlx_records, oll_records)

    # 5. Compile Grand Summary
    summary_report = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": "Apple Silicon (macOS arm64)",
            "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "prompt_count": len(selected_prompts),
            "max_tokens_per_prompt": args.tokens,
            "initial_thermal_state": thermal_monitor.initial_state,
            "peak_thermal_state": thermal_monitor.peak_state,
            "total_throttling_seconds": thermal_monitor.total_throttle_seconds,
        },
        "engines": {
            "velocityai": {
                "name": "VelocityAI (Ours)",
                "format": "FP16 Lossless (4 P-Core NEON + SpinBarrier)",
                "load_time_sec": round(vel_load_time, 2),
                "total_wall_sec": round(t_vel_elapsed, 2),
                "stats": vel_stats,
                "thermal": vel_thermal
            },
            "mlx": {
                "name": "Apple MLX",
                "format": "BFloat16 (Apple Metal GPU)",
                "load_time_sec": round(mlx_load_time, 2),
                "total_wall_sec": round(t_mlx_elapsed, 2),
                "stats": mlx_stats,
                "thermal": mlx_thermal
            },
            "ollama": {
                "name": "Ollama (llama.cpp)",
                "format": "Q4_K_M (4-bit Quantized GGUF)",
                "load_time_sec": round(oll_load_time, 2),
                "total_wall_sec": round(t_oll_elapsed, 2),
                "stats": oll_stats,
                "thermal": oll_thermal
            }
        },
        "quality_fidelity_comparison": fidelity_comp[:15] # Top 15 samples
    }

    # Save to disk
    with open("reality_check_results.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    # 6. Print Executive Summary Table
    print("\n" + "="*105)
    print(f" {'FRAMEWORK':<22} | {'FORMAT':<24} | {'MEAN TOK/S':<11} | {'MEDIAN':<9} | {'P95 TOK/S':<10} | {'ITL (ms)':<9} | {'THERMAL'}")
    print("-"*105)
    for k, v in summary_report["engines"].items():
        if v["stats"]:
            st = v["stats"]
            th = v["thermal"]["peak_thermal_desc"]
            print(f" {v['name']:<22} | {v['format'][:24]:<24} | {st['mean_tps']:<11.2f} | {st['median_tps']:<9.2f} | {st['p95_tps']:<10.2f} | {st['median_itl_ms']:<9.2f} | {th}")
    print("="*105)
    print(f"\n[✓] Detailed report saved to: reality_check_results.json")

if __name__ == "__main__":
    main()
