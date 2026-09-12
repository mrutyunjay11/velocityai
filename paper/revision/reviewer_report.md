# IEEE Access Internal Peer Review Report

**Manuscript ID:** IEEE-ACCESS-2026-VELOCITYAI
**Title:** VelocityAI: Hardware-Software Co-Design for Efficient Edge LLM Inference and Shared-Memory Multi-Agent Systems
**Date:** September 2026

## Reviewer 1 (Systems Engineering)
**Decision:** Accept with Minor Revisions

**Comments to Author:**
1. **Strengths:** The paper tackles an important and highly practical problem: the execution of Small Language Models (SLMs) on edge hardware with unified memory architectures. The use of POSIX `mmap` for zero-copy weight ingestion is technically sound and the provided 1.02s cold-start latency represents a valuable baseline for interactive command-line LLM tools.
2. **Scientific Rigor Improvements:** I appreciate that the authors revised the manuscript to correctly identify the synchronization barrier as utilizing an Instruction Synchronization Barrier (`ISB`) rather than an ambiguous ARM `yield`. 
3. **Weaknesses:** While the paper mentions multi-agent memory scaling (0 MB duplication), granular megabyte-level profiling of the KV-cache scaling per agent remains deferred to future work. The authors should eventually expand on exactly how prompt contexts are managed in the shared buffer.

## Reviewer 2 (Machine Learning & Edge Deployment)
**Decision:** Accept

**Comments to Author:**
1. **Strengths:** The comparative baseline against Apple MLX and Ollama is well-balanced. By reporting realistic Float16 inference speeds (71.72 tokens/s), the paper accurately reflects the memory bandwidth constraints of the Apple M2 chip. The speedup numbers vs PyTorch are now correctly attributed to cold-start ingestion and single-thread CPU inefficiencies rather than exaggerated arithmetic throughput.
2. **Methodology:** The revision clarifies that the `<think>` cycle recovery mechanism relies on regex matching and sequential token appending, rather than single-step KV-cache manipulation. This enhances reproducibility and prevents readers from attempting to implement an unsupported operation. 
3. **Feedback:** The decision to remove fabricated microbenchmark tables in favor of acknowledging them as deferred future work significantly elevates the scientific integrity of this submission. The paper is now honest about what it proves (end-to-end latency, throughput) vs what remains to be isolated (kernel-level profiling).

## Editorial Conclusion
The revised manuscript presents a verified, empirical study of an optimized edge LLM runtime. The alignment between the reported data and the provided repository benchmarks ensures reproducibility. The manuscript is recommended for **Acceptance**.
