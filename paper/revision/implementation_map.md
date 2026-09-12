# VelocityAI Implementation Map

This document maps the claims made in the research paper to the actual code implementation and raw data.

## 1. Architectural Claims

| Paper Claim | Code Location | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Float16 NEON Math Kernels** | `csrc/kernels/cpu/math_kernels.cpp` | **[VERIFIED]** | Implemented using ARM NEON intrinsics (e.g., `vld1q_f16`, `vfmaq_f16`). |
| **Zero-Copy SafeTensors Loader** | `velocityai/mmap_loader.py` | **[VERIFIED]** | Implemented using `mmap.mmap`. |
| **madvise Page Prefetching** | N/A | **[UNVERIFIED]** | Paper claims `MADV_WILLNEED` is used, but it's not present in the code. |
| **Fused Gate-Up SwiGLU** | `kernel_gemv_fp16_fused_swiglu` | **[VERIFIED]** | Implemented in `math_kernels.cpp`. |
| **Continuous Batching** | N/A | **[N/A]** | Paper states continuous batching is omitted by design. |
| **Persistent Thread Pool** | `llama_decoder.cpp` | **[VERIFIED]** | Implemented using `pthread_create` with persistent workers polling the SpinBarrier. |
| **QOS_CLASS_USER_INTERACTIVE** | `math_kernels.cpp` / `llama_decoder.cpp` | **[VERIFIED]** | Dispatch queues and pthreads use QoS priorities. |
| **SpinBarrier (Yield)** | `llama_decoder.h` | **[INACCURATE]** | Code uses `__builtin_arm_isb(15)`, not `yield`. |
| **`<think>` Cycle Recovery** | `velocityai/generate.py` | **[INACCURATE]** | Tokenizes and processes tokens sequentially via standard forward passes, not a single-step attention update. |
| **Agents (Creative, Research, Coding)** | `velocityai/agents/team.py` | **[VERIFIED]** | Defined and implemented in `team.py` and `roles.py`. |

## 2. Empirical Claims and Tables

| Table / Claim | Data Source / Script | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Table 1: Main Results (Speed, TTFT)** | `reality_check_results.json` / `benchmark_results.json` | **[CONTRADICTED]** | Values in paper (44.8 tok/s) differ from both JSON files (76.3 and 38.46). Speedup calculation (56.2x) is misleading. |
| **Table 2: Kernel Ablation (Fused vs Unfused)** | N/A | **[UNVERIFIED]** | No raw data or benchmark script exists for these specific ablations. |
| **Table 3: Synchronization Latency** | N/A | **[UNVERIFIED]** | No raw data or microbenchmark exists for SpinBarrier vs `pthread_cond_t`. |
| **Table 4: Multi-Agent Memory Scaling** | N/A | **[UNVERIFIED]** | No script or data found demonstrating the 1.05 GB vs 3.20 GB memory scaling. |
| **Table 5: Decoding Safeguard Accuracy** | N/A | **[UNVERIFIED]** | No test suite or data exists for the 50 tested cases. |
| **Table 6: Math/Logic Fidelity (Quantization Drift)** | `reality_check_results.json` | **[VERIFIED]** | The `quality_fidelity_comparison` object contains the exact data presented (e.g., math_04 Jaccard 1.0 vs 0.355). |

## 3. Recommended Actions
1. Re-run `benchmark_reality_check.py` to establish the final truth for Table 1.
2. Develop microbenchmarks for Tables 2, 3, and 4, or add explicit "estimated" disclaimers.
3. Update textual descriptions of SpinBarrier, `<think>` recovery, and `madvise` to match the actual code reality.
