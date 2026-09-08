# VelocityAI ⚡

High-Level to Low-Level AI Acceleration Engine.

VelocityAI compiles and optimizes high-level Python code into high-performance, low-level machine code for CPU, CUDA, and Apple Metal.

## Phase 1 Architecture
- **C++ Tensor Core**: Custom memory pool with 64-byte alignment (AVX-512 / ARM NEON ready).
- **Zero-Copy Interoperability**: Direct zero-copy sharing with NumPy using the Python buffer protocol.
- **Strided Layout & Views**: Zero-copy `reshape()`, `transpose()`, `permute()`, and `slice()`.
- **Tiled Cache GEMM**: Matrix multiplication optimized with cache tiling and vectorization.

## Quick Start

```python
import velocityai as vai

# Create tensors
x = vai.tensor([[1.0, 2.0], [3.0, 4.0]])
w = vai.tensor([[0.5, 0.1], [0.3, 0.7]])

# Operations execute in high-performance C++ backend
y = vai.matmul(x, w)
z = vai.relu(y)

# Zero-copy conversion to NumPy
arr = z.numpy()
print(arr)
```
