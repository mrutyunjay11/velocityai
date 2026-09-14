#include <stdint.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#if defined(__APPLE__) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Helper: Apply RoPE rotation to a single vector of length head_dim
static inline void apply_rope_inplace(float* vec, const float* cos, const float* sin, int64_t head_dim) {
    int64_t half = head_dim / 2;
#if defined(__APPLE__) && defined(__aarch64__)
    for (int64_t i = 0; i <= half - 4; i += 4) {
        float32x4_t x1 = vld1q_f32(vec + i);
        float32x4_t x2 = vld1q_f32(vec + i + half);
        float32x4_t c = vld1q_f32(cos + i);
        float32x4_t s = vld1q_f32(sin + i);
        float32x4_t r1 = vsubq_f32(vmulq_f32(x1, c), vmulq_f32(x2, s));
        float32x4_t r2 = vaddq_f32(vmulq_f32(x2, c), vmulq_f32(x1, s));
        vst1q_f32(vec + i, r1);
        vst1q_f32(vec + i + half, r2);
    }
#else
    for (int64_t i = 0; i < half; ++i) {
        float x1 = vec[i];
        float x2 = vec[i + half];
        float c = cos[i];
        float s = sin[i];
        vec[i] = x1 * c - x2 * s;
        vec[i + half] = x2 * c + x1 * s;
    }
#endif
}

// Fused decode attention for single query token (T=1)
// Q: (num_heads, head_dim)
// K: (num_kv_heads, head_dim)
// V: (num_kv_heads, head_dim)
// K_cache: (max_seq_len, num_kv_heads, head_dim)
// V_cache: (max_seq_len, num_kv_heads, head_dim)
// out_context: (num_heads * head_dim)
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
) {
    // 1. Apply RoPE in-place to all query heads
    for (int64_t h = 0; h < num_heads; ++h) {
        apply_rope_inplace(Q + h * head_dim, cos, sin, head_dim);
    }
    
    // 2. Apply RoPE in-place to all key heads
    for (int64_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
        apply_rope_inplace(K + kv_h * head_dim, cos, sin, head_dim);
    }
    
    // 3. Store new K and V into cache at start_pos
    int64_t kv_step_stride = num_kv_heads * head_dim;
    std::memcpy(K_cache + start_pos * kv_step_stride, K, kv_step_stride * sizeof(float));
    std::memcpy(V_cache + start_pos * kv_step_stride, V, kv_step_stride * sizeof(float));
    
    int64_t total_tokens = start_pos + 1;
    int64_t kv_groups = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Dynamically allocate attention scores to avoid stack overflow for long sequences (>4096)
    std::unique_ptr<float[]> scores_buf(new float[total_tokens]);
    float* scores = scores_buf.get();
    
    // 4. Compute attention for each head
    for (int64_t h = 0; h < num_heads; ++h) {
        int64_t kv_h = h / kv_groups;
        const float* q_ptr = Q + h * head_dim;
        float* out_h = out_context + h * head_dim;
        
        // Zero output vector for this head
        std::memset(out_h, 0, head_dim * sizeof(float));
        
        // Dot product with all cached keys
        float max_score = -1e30f;
        for (int64_t t = 0; t < total_tokens; ++t) {
            const float* k_ptr = K_cache + t * kv_step_stride + kv_h * head_dim;
            
#if defined(__APPLE__) && defined(__aarch64__)
            float32x4_t vd0 = vmulq_f32(vld1q_f32(q_ptr), vld1q_f32(k_ptr));
            float32x4_t vd1 = vmulq_f32(vld1q_f32(q_ptr + 4), vld1q_f32(k_ptr + 4));
            float32x4_t vd2 = vmulq_f32(vld1q_f32(q_ptr + 8), vld1q_f32(k_ptr + 8));
            float32x4_t vd3 = vmulq_f32(vld1q_f32(q_ptr + 12), vld1q_f32(k_ptr + 12));
            for (int64_t d = 16; d <= head_dim - 16; d += 16) {
                vd0 = vfmaq_f32(vd0, vld1q_f32(q_ptr + d), vld1q_f32(k_ptr + d));
                vd1 = vfmaq_f32(vd1, vld1q_f32(q_ptr + d + 4), vld1q_f32(k_ptr + d + 4));
                vd2 = vfmaq_f32(vd2, vld1q_f32(q_ptr + d + 8), vld1q_f32(k_ptr + d + 8));
                vd3 = vfmaq_f32(vd3, vld1q_f32(q_ptr + d + 12), vld1q_f32(k_ptr + d + 12));
            }
            float dot = vaddvq_f32(vaddq_f32(vaddq_f32(vd0, vd1), vaddq_f32(vd2, vd3)));
            for (int64_t d = head_dim & ~15; d < head_dim; ++d) {
                dot += q_ptr[d] * k_ptr[d];
            }
#else
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot += q_ptr[d] * k_ptr[d];
            }
#endif
            float s = dot * scale;
            scores[t] = s;
            if (s > max_score) max_score = s;
        }
        
        // Softmax
        float sum_exp = 0.0f;
        for (int64_t t = 0; t < total_tokens; ++t) {
            float exp_val = std::exp(scores[t] - max_score);
            scores[t] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int64_t t = 0; t < total_tokens; ++t) {
            scores[t] *= inv_sum;
        }
        
        // Weighted sum over V cache
        for (int64_t t = 0; t < total_tokens; ++t) {
            float w = scores[t];
            const float* v_ptr = V_cache + t * kv_step_stride + kv_h * head_dim;
#if defined(__APPLE__) && defined(__aarch64__)
            float32x4_t vw = vdupq_n_f32(w);
            for (int64_t d = 0; d <= head_dim - 16; d += 16) {
                vst1q_f32(out_h + d, vfmaq_f32(vld1q_f32(out_h + d), vw, vld1q_f32(v_ptr + d)));
                vst1q_f32(out_h + d + 4, vfmaq_f32(vld1q_f32(out_h + d + 4), vw, vld1q_f32(v_ptr + d + 4)));
                vst1q_f32(out_h + d + 8, vfmaq_f32(vld1q_f32(out_h + d + 8), vw, vld1q_f32(v_ptr + d + 8)));
                vst1q_f32(out_h + d + 12, vfmaq_f32(vld1q_f32(out_h + d + 12), vw, vld1q_f32(v_ptr + d + 12)));
            }
            for (int64_t d = head_dim & ~15; d < head_dim; ++d) {
                out_h[d] += w * v_ptr[d];
            }
#else
            for (int64_t d = 0; d < head_dim; ++d) {
                out_h[d] += w * v_ptr[d];
            }
#endif
        }
    }
}

// Batched causal attention for prefill
void kernel_attention_prefill_f32(
    float* QKV,
    float* K_cache,
    float* V_cache,
    const float* cos_table,
    const float* sin_table,
    float* out_context,
    int64_t N,
    int64_t start_pos,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t max_seq_len
) {
    int64_t q_stride = num_heads * head_dim;
    int64_t kv_stride = num_kv_heads * head_dim;
    int64_t qkv_stride = q_stride + 2 * kv_stride;
    
    // 1. Apply RoPE in-place and write to cache
    for (int64_t i = 0; i < N; ++i) {
        int64_t pos = start_pos + i;
        const float* cos_p = cos_table + pos * head_dim;
        const float* sin_p = sin_table + pos * head_dim;
        
        float* q_i = QKV + i * qkv_stride;
        float* k_i = QKV + i * qkv_stride + q_stride;
        float* v_i = QKV + i * qkv_stride + q_stride + kv_stride;
        
        // RoPE for Q
        for (int64_t h = 0; h < num_heads; ++h) {
            apply_rope_inplace(q_i + h * head_dim, cos_p, sin_p, head_dim);
        }
        
        // RoPE for K
        for (int64_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
            apply_rope_inplace(k_i + kv_h * head_dim, cos_p, sin_p, head_dim);
        }
        
        // Write K, V into cache at pos
        std::memcpy(K_cache + pos * kv_stride, k_i, kv_stride * sizeof(float));
        std::memcpy(V_cache + pos * kv_stride, v_i, kv_stride * sizeof(float));
    }
    
    int64_t total_tokens = start_pos + N;
    int64_t kv_groups = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Process each head
    std::unique_ptr<float[]> scores_buf(new float[total_tokens]);
    float* scores = scores_buf.get();
    
    for (int64_t h = 0; h < num_heads; ++h) {
        int64_t kv_h = h / kv_groups;
        
        for (int64_t i = 0; i < N; ++i) {
            float* q_ptr = QKV + i * qkv_stride + h * head_dim;
            float* out_ptr = out_context + i * q_stride + h * head_dim;
            std::memset(out_ptr, 0, head_dim * sizeof(float));
            
            int64_t current_pos = start_pos + i;
            float max_score = -INFINITY;
            
            // Compute scores for j in [0, current_pos] (causal mask)
            for (int64_t j = 0; j <= current_pos; ++j) {
                const float* k_ptr = K_cache + j * kv_stride + kv_h * head_dim;
                float dot = 0.0f;
#if defined(__APPLE__) && defined(__aarch64__)
                float32x4_t vd0 = vdupq_n_f32(0.0f), vd1 = vdupq_n_f32(0.0f);
                float32x4_t vd2 = vdupq_n_f32(0.0f), vd3 = vdupq_n_f32(0.0f);
                for (int64_t d = 0; d <= head_dim - 16; d += 16) {
                    vd0 = vfmaq_f32(vd0, vld1q_f32(q_ptr + d), vld1q_f32(k_ptr + d));
                    vd1 = vfmaq_f32(vd1, vld1q_f32(q_ptr + d + 4), vld1q_f32(k_ptr + d + 4));
                    vd2 = vfmaq_f32(vd2, vld1q_f32(q_ptr + d + 8), vld1q_f32(k_ptr + d + 8));
                    vd3 = vfmaq_f32(vd3, vld1q_f32(q_ptr + d + 12), vld1q_f32(k_ptr + d + 12));
                }
                dot = vaddvq_f32(vaddq_f32(vaddq_f32(vd0, vd1), vaddq_f32(vd2, vd3)));
                for (int64_t d = head_dim & ~15; d < head_dim; ++d) {
                    dot += q_ptr[d] * k_ptr[d];
                }
#else
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q_ptr[d] * k_ptr[d];
                }
#endif
                float s = dot * scale;
                scores[j] = s;
                if (s > max_score) max_score = s;
            }
            
            // Softmax over j in [0, current_pos]
            float sum_exp = 0.0f;
            for (int64_t j = 0; j <= current_pos; ++j) {
                float exp_val = std::exp(scores[j] - max_score);
                scores[j] = exp_val;
                sum_exp += exp_val;
            }
            float inv_sum = 1.0f / sum_exp;
            for (int64_t j = 0; j <= current_pos; ++j) {
                scores[j] *= inv_sum;
            }
            
            // Weighted sum over V cache
            for (int64_t j = 0; j <= current_pos; ++j) {
                float w = scores[j];
                const float* v_ptr = V_cache + j * kv_stride + kv_h * head_dim;
#if defined(__APPLE__) && defined(__aarch64__)
                float32x4_t vw = vdupq_n_f32(w);
                for (int64_t d = 0; d <= head_dim - 16; d += 16) {
                    vst1q_f32(out_ptr + d, vfmaq_f32(vld1q_f32(out_ptr + d), vw, vld1q_f32(v_ptr + d)));
                    vst1q_f32(out_ptr + d + 4, vfmaq_f32(vld1q_f32(out_ptr + d + 4), vw, vld1q_f32(v_ptr + d + 4)));
                    vst1q_f32(out_ptr + d + 8, vfmaq_f32(vld1q_f32(out_ptr + d + 8), vw, vld1q_f32(v_ptr + d + 8)));
                    vst1q_f32(out_ptr + d + 12, vfmaq_f32(vld1q_f32(out_ptr + d + 12), vw, vld1q_f32(v_ptr + d + 12)));
                }
                for (int64_t d = head_dim & ~15; d < head_dim; ++d) {
                    out_ptr[d] += w * v_ptr[d];
                }
#else
                for (int64_t d = 0; d < head_dim; ++d) {
                    out_ptr[d] += w * v_ptr[d];
                }
#endif
            }
        }
    }
}


void kernel_attention_f32(const float* Q, const float* K, const float* V, float* out, 
                          int64_t batch, int64_t num_heads, int64_t seq_len, int64_t head_dim) {
}

#ifdef __cplusplus
}
#endif
