#ifndef VAI_MATH_KERNELS_H
#define VAI_MATH_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void kernel_add_f32(const float* a, const float* b, float* out, int64_t n);
void kernel_sub_f32(const float* a, const float* b, float* out, int64_t n);
void kernel_mul_f32(const float* a, const float* b, float* out, int64_t n);
void kernel_div_f32(const float* a, const float* b, float* out, int64_t n);

void kernel_add_scalar_f32(const float* a, float val, float* out, int64_t n);
void kernel_mul_scalar_f32(const float* a, float val, float* out, int64_t n);

void kernel_matmul_f32(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N);
void kernel_matmul_transposed_f32(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N);

void kernel_relu_f32(const float* in, float* out, int64_t n);

void kernel_gemv_fp16(const float* x, const void* W, float* out, int64_t K, int64_t N);
void kernel_gemv_fp16_acc(const float* x, const void* W, float* out, int64_t K, int64_t N);
void kernel_gemm_fp16(const float* x, const void* W, float* out, int64_t M, int64_t K, int64_t N, float** f32_cache_ptr = nullptr);
void kernel_gemv_fp16_fused_swiglu(const float* x, const void* W_gate_up, float* hidden_out, int64_t K, int64_t intermediate);

void kernel_gemv_f32(const float* x, const float* W, float* out, int64_t K, int64_t N);
void kernel_gemv_f32_fused_swiglu(const float* x, const float* W_gate_up, float* hidden_out, int64_t K, int64_t intermediate);

#ifdef __cplusplus
}
#endif

#endif
