"""Example 01: VelocityAI Tensor Basics and NumPy Interoperability."""

import velocityai as vai
import numpy as np
import time

def main():
    print("=== VelocityAI Phase 1: Tensor Engine ===")

    # 1. Tensor creation
    x = vai.tensor([[1.0, 2.0], [3.0, 4.0]])
    w = vai.tensor([[0.5, 0.1], [0.3, 0.7]])
    print(f"Tensor x:\n{x}")
    print(f"Tensor w:\n{w}")

    # 2. Arithmetic & Matrix Multiplication in C++ backend
    y = vai.matmul(x, w)
    print(f"\nMatrix Multiplication (x @ w):\n{y}")

    # 3. Activation
    z = vai.relu(y - 1.0)
    print(f"\nReLU(y - 1.0):\n{z}")

    # 4. Zero-copy NumPy Interoperability
    arr = y.numpy()
    print(f"\nZero-copy NumPy ndarray:\n{arr}")
    print(f"NumPy array type: {type(arr)}, dtype: {arr.dtype}")

    # 5. Views and Transposition (zero-copy)
    t = vai.randn(10, 5)
    t_T = t.T
    print(f"\nOriginal shape: {t.shape}, strides: {t.strides}, is_contiguous: {t.is_contiguous}")
    print(f"Transposed shape: {t_T.shape}, strides: {t_T.strides}, is_contiguous: {t_T.is_contiguous}")

    # 6. Memory Pool Performance
    print(f"\nAllocated memory: {vai.memory.allocated_bytes() / (1024 * 1024):.2f} MB")
    print(f"Peak memory:      {vai.memory.peak_bytes() / (1024 * 1024):.2f} MB")

    # 7. Speed benchmark against NumPy
    N = 1024
    print(f"\nBenchmarking {N}x{N} Matrix Multiplication...")
    a_np = np.random.randn(N, N).astype(np.float32)
    b_np = np.random.randn(N, N).astype(np.float32)

    a_vai = vai.from_numpy(a_np)
    b_vai = vai.from_numpy(b_np)

    t0 = time.perf_counter()
    c_vai = vai.matmul(a_vai, b_vai)
    t_vai = time.perf_counter() - t0

    t0 = time.perf_counter()
    c_np = a_np @ b_np
    t_np = time.perf_counter() - t0

    print(f"VelocityAI C++ Tiled MatMul: {t_vai * 1000:.2f} ms")
    print(f"NumPy OpenBLAS/Accelerate:   {t_np * 1000:.2f} ms")
    diff = np.max(np.abs(c_vai.numpy() - c_np))
    print(f"Max numerical difference:    {diff:.6e}")

    vai.memory.empty_cache()
    print("\nMemory cache cleared. Phase 1 demo completed successfully!")

if __name__ == "__main__":
    main()
