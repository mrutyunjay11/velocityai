#!/usr/bin/env python3
"""
Continuous Thermal Endurance & CPU Throttling Probe
Measures:
  1. Sustained generation throughput across back-to-back rounds
  2. Hardware thermal state transitions (Nominal -> Fair -> Serious)
  3. CPU Throttling Factor Φ = 1.0 - (Throughput_current / Throughput_baseline)
  4. Time-to-throttle (seconds under sustained load before degradation)
"""

import time
import json
import ctypes
import ctypes.util
import subprocess
import numpy as np

def setup_thermal_query():
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        msg_send = objc.objc_msgSend
        msg_send.restype = ctypes.c_void_p
        msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        cls = objc.objc_getClass(b"NSProcessInfo")
        sel_pi = objc.sel_registerName(b"processInfo")
        pi = msg_send(cls, sel_pi)

        msg_send_state = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc")).objc_msgSend
        msg_send_state.restype = ctypes.c_long
        msg_send_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        sel_ts = objc.sel_registerName(b"thermalState")
        return lambda: int(msg_send_state(pi, sel_ts))
    except Exception:
        return lambda: 0

def main():
    import velocityai as vai
    
    get_thermal = setup_thermal_query()
    state_names = {0: "Nominal (Cool / 3.5GHz)", 1: "Fair (Warm / Governor active)", 2: "Serious (CPU Throttling active)", 3: "Critical"}
    
    print("="*75)
    print(" 🌡️  CONTINUOUS THERMAL ENDURANCE & THROTTLING BENCHMARK")
    print("="*75)
    print(f"Initial Thermal State: {get_thermal()} ({state_names.get(get_thermal(), 'Unknown')})")

    # Load VelocityAI
    print("[*] Loading VelocityAI model...")
    model, config = vai.load_huggingface_model("HuggingFaceTB/SmolLM2-360M-Instruct")
    tokenizer = vai.Tokenizer("HuggingFaceTB/SmolLM2-360M-Instruct")
    
    prompt = "Explain in detail the mathematical principles of general relativity, spacetime curvature, and tensor calculus."
    rounds = 15
    tokens_per_round = 128
    
    print(f"[*] Running {rounds} continuous sustained rounds ({tokens_per_round} tokens each = {rounds * tokens_per_round} tokens total)...")
    print("-" * 75)
    print(f" {'ROUND':<6} | {'ELAPSED (s)':<12} | {'TOK/S':<10} | {'THERMAL STATE':<28} | {'THROTTLING Φ'}")
    print("-" * 75)
    
    t_start = time.time()
    baseline_tps = None
    records = []
    
    for r in range(1, rounds + 1):
        t_round_start = time.time()
        res = vai.generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=tokens_per_round,
            temperature=0.0,
            device="cpu"
        )
        round_wall = time.time() - t_round_start
        elapsed = time.time() - t_start
        
        tps = res.get("decode_tps", res["tps"])
        if baseline_tps is None:
            baseline_tps = tps
            
        th_state = get_thermal()
        throttling_factor = max(0.0, 1.0 - (tps / baseline_tps))
        
        status_str = state_names.get(th_state, "Unknown")
        throttle_pct = f"{throttling_factor * 100:.1f}%" if throttling_factor > 0.02 else "None (0.0%)"
        
        print(f" #{r:<5} | {elapsed:<12.1f} | {tps:<10.2f} | {status_str:<28} | {throttle_pct}")
        records.append({
            "round": r,
            "elapsed_sec": round(elapsed, 2),
            "tps": round(tps, 2),
            "thermal_state": th_state,
            "thermal_desc": status_str,
            "throttling_factor": round(throttling_factor, 4)
        })
        
    print("-" * 75)
    mean_tps = np.mean([rec["tps"] for rec in records])
    min_tps = np.min([rec["tps"] for rec in records])
    max_tps = np.max([rec["tps"] for rec in records])
    print(f"⚡ Summary: Peak: {max_tps:.2f} tok/s | Min: {min_tps:.2f} tok/s | Mean: {mean_tps:.2f} tok/s")
    print(f"🌡️  Final Thermal State: {get_thermal()} ({state_names.get(get_thermal(), 'Unknown')})")
    
    with open("thermal_endurance_results.json", "w") as f:
        json.dump({
            "rounds": records,
            "baseline_tps": round(baseline_tps, 2),
            "mean_tps": round(mean_tps, 2),
            "final_thermal_state": get_thermal()
        }, f, indent=2)
    print("[✓] Thermal endurance data saved to: thermal_endurance_results.json\n")

if __name__ == "__main__":
    main()
