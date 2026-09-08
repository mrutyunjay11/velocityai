#ifndef VAI_CPU_KERNELS_H
#define VAI_CPU_KERNELS_H

#include "csrc/kernels/cpu/math_kernels.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void kernel_sum_f32(const float* in, float* out, int64_t numel);

void kernel_softmax_f32(const float* in, float* out, int64_t rows, int64_t cols);

void kernel_layernorm_f32(const float* in, const float* weight, const float* bias, 
                          float* out, int64_t rows, int64_t cols, float eps);

void kernel_rmsnorm_f32(const float* in, const float* weight, float* out, int64_t rows, int64_t cols, float eps);
void kernel_rmsnorm_to_fp16(const float* in, const float* weight, void* out_fp16, int64_t cols, float eps);

void kernel_swiglu_f32(const float* gate, const float* up, float* out, int64_t n);

void kernel_attention_f32(const float* Q, const float* K, const float* V, float* out, 
                          int64_t batch, int64_t num_heads, int64_t seq_len, int64_t head_dim);

void kernel_attention_decode_f32(
    float* Q,
    float* K,
    const float* V,
    float* K_cache,
    float* V_cache,
    const float* cos,
    const float* sin,
    float* out_context,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t max_seq_len,
    int64_t start_pos
);

void kernel_conv2d_f32(const float* in, const float* weight, const float* bias, float* out,
                       int64_t batch, int64_t in_ch, int64_t in_h, int64_t in_w,
                       int64_t out_ch, int64_t k_h, int64_t k_w,
                       int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w);

void kernel_cross_entropy_f32(const float* logits, const int64_t* targets, float* out_loss, int64_t batch, int64_t classes);

void kernel_embedding_f32(const int64_t* indices, const float* weight, float* out, 
                          int64_t num_indices, int64_t embedding_dim);

#ifdef __cplusplus
}
#endif

#endif
