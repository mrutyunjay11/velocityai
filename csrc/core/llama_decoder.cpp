#include "csrc/core/llama_decoder.h"
#include "csrc/kernels/cpu/cpu_kernels.h"
#include "csrc/kernels/cpu/math_kernels.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <utility>

#if defined(__APPLE__) && defined(__aarch64__)
#include <arm_neon.h>
#include <pthread/qos.h>
#endif

namespace velocityai {

static inline void float_to_fp16_neon(const float* src, __fp16* dst, int64_t count) {
#if defined(__APPLE__) && defined(__aarch64__)
    int64_t i = 0;
    for (; i <= count - 8; i += 8) {
        float32x4_t v0 = vld1q_f32(src + i);
        float32x4_t v1 = vld1q_f32(src + i + 4);
        float16x4_t h0 = vcvt_f16_f32(v0);
        float16x4_t h1 = vcvt_f16_f32(v1);
        vst1q_f16(dst + i, vcombine_f16(h0, h1));
    }
    for (; i < count; ++i) dst[i] = static_cast<__fp16>(src[i]);
#else
    for (int64_t i = 0; i < count; ++i) dst[i] = static_cast<__fp16>(src[i]);
#endif
}

FastLlamaDecoder::FastLlamaDecoder(
    int64_t vocab_size,
    int64_t dim,
    int64_t num_layers,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t intermediate_size,
    int64_t max_seq_len,
    float eps,
    float rope_theta,
    const Tensor& embed_tokens_weight,
    const Tensor& norm_weight,
    const Tensor& lm_head_weight
) : vocab_size_(vocab_size),
    dim_(dim),
    num_layers_(num_layers),
    num_heads_(num_heads),
    num_kv_heads_(num_kv_heads),
    head_dim_(head_dim),
    intermediate_size_(intermediate_size),
    max_seq_len_(max_seq_len),
    eps_(eps),
    rope_theta_(rope_theta),
    num_threads_(4),
    barrier_start_(4),
    barrier_end_(4),
    curr_task_(TASK_NONE)
{
    pinned_tensors_.push_back(embed_tokens_weight.contiguous());
    pinned_tensors_.push_back(norm_weight.contiguous());
    pinned_tensors_.push_back(lm_head_weight.contiguous());

    embed_tokens_weight_ = pinned_tensors_[0].data_ptr();
    norm_weight_ = pinned_tensors_[1].data_ptr<float>();
    lm_head_weight_ = pinned_tensors_[2].data_ptr();

    // Precompute RoPE tables
    cos_table_.resize(max_seq_len_ * head_dim_);
    sin_table_.resize(max_seq_len_ * head_dim_);
    int64_t half = head_dim_ / 2;
    for (int64_t pos = 0; pos < max_seq_len_; ++pos) {
        for (int64_t i = 0; i < half; ++i) {
            float inv_freq = 1.0f / std::pow(rope_theta_, static_cast<float>(2 * i) / static_cast<float>(head_dim_));
            float freq = static_cast<float>(pos) * inv_freq;
            float c = std::cos(freq);
            float s = std::sin(freq);
            cos_table_[pos * head_dim_ + i] = c;
            cos_table_[pos * head_dim_ + i + half] = c;
            sin_table_[pos * head_dim_ + i] = s;
            sin_table_[pos * head_dim_ + i + half] = s;
        }
    }

    // Allocate scratch buffers
    int64_t qkv_dim = (num_heads_ + 2 * num_kv_heads_) * head_dim_;
    h_buf_.resize(dim_);
    norm_buf_.resize(dim_);
    qkv_buf_.resize(qkv_dim);
    attn_out_buf_.resize(num_heads_ * head_dim_);
    norm2_buf_.resize(dim_);
    swiglu_buf_.resize(intermediate_size_);
    logits_buf_.resize(vocab_size_);
    x16_buf_.resize(std::max(dim_, intermediate_size_));

    // Spawn 3 worker threads pinned to interactive QoS
    for (int i = 1; i < num_threads_; ++i) {
        auto* p = new std::pair<FastLlamaDecoder*, int>(this, i);
        pthread_t th;
        pthread_create(&th, nullptr, worker_thread_func, p);
        worker_handles_.push_back(th);
    }
#if defined(__APPLE__) && defined(__aarch64__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

FastLlamaDecoder::~FastLlamaDecoder() {
    curr_task_.store(TASK_STOP, std::memory_order_relaxed);
    barrier_start_.wait(0);
    for (pthread_t th : worker_handles_) {
        pthread_join(th, nullptr);
    }
}

void FastLlamaDecoder::add_layer(
    const Tensor& input_layernorm_weight,
    const Tensor& qkv_weight,
    const Tensor& o_weight,
    const Tensor& post_attn_layernorm_weight,
    const Tensor& gate_up_weight,
    const Tensor& down_weight,
    Tensor& k_cache,
    Tensor& v_cache,
    const Tensor* qkv_bias
) {
    Tensor in_norm_c = input_layernorm_weight.contiguous();
    Tensor qkv_c = qkv_weight.contiguous();
    Tensor o_c = o_weight.contiguous();
    Tensor post_norm_c = post_attn_layernorm_weight.contiguous();
    Tensor gu_c = gate_up_weight.contiguous();
    Tensor down_c = down_weight.contiguous();
    Tensor kc_c = k_cache.contiguous();
    Tensor vc_c = v_cache.contiguous();

    pinned_tensors_.push_back(in_norm_c);
    pinned_tensors_.push_back(qkv_c);
    pinned_tensors_.push_back(o_c);
    pinned_tensors_.push_back(post_norm_c);
    pinned_tensors_.push_back(gu_c);
    pinned_tensors_.push_back(down_c);
    pinned_tensors_.push_back(kc_c);
    pinned_tensors_.push_back(vc_c);

    FastLayerWeights lw;
    lw.input_layernorm_weight = in_norm_c.data_ptr<float>();
    lw.qkv_weight = qkv_c.data_ptr();
    lw.o_weight = o_c.data_ptr();
    lw.post_attn_layernorm_weight = post_norm_c.data_ptr<float>();
    lw.gate_up_weight = gu_c.data_ptr();
    lw.down_weight = down_c.data_ptr();
    lw.k_cache = kc_c.data_ptr<float>();
    lw.v_cache = vc_c.data_ptr<float>();
    
    if (qkv_bias != nullptr) {
        Tensor qb_c = qkv_bias->contiguous();
        pinned_tensors_.push_back(qb_c);
        lw.qkv_bias = qb_c.data_ptr<float>();
    } else {
        lw.qkv_bias = nullptr;
    }

    layers_.push_back(lw);
}

void* FastLlamaDecoder::worker_thread_func(void* arg) {
    auto* p = static_cast<std::pair<FastLlamaDecoder*, int>*>(arg);
    FastLlamaDecoder* self = p->first;
    int tid = p->second;
    delete p;

#if defined(__APPLE__) && defined(__aarch64__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    while (true) {
        self->barrier_start_.wait(tid);
        DecoderTaskType task = self->curr_task_.load(std::memory_order_relaxed);
        if (task == TASK_STOP) break;

        self->execute_worker_slice(tid);
        self->barrier_end_.wait(tid);
    }
    return nullptr;
}

void FastLlamaDecoder::execute_worker_slice(int tid) {
    int64_t chunk = curr_N_ / num_threads_;
    int64_t start = tid * chunk;
    int64_t end = (tid == num_threads_ - 1) ? curr_N_ : (tid + 1) * chunk;

    const __fp16* x16 = reinterpret_cast<const __fp16*>(x16_buf_.data());
    const __fp16* W16 = reinterpret_cast<const __fp16*>(curr_weight_);
    float* out = curr_out_;
    int64_t K = curr_K_;
    DecoderTaskType task = curr_task_.load(std::memory_order_relaxed);

    if (task == TASK_QKV || task == TASK_O || task == TASK_DOWN) {
        bool acc = (task != TASK_QKV);
        int64_t j = start;
        for (; j <= end - 2; j += 2) {
            const __fp16* w_row0 = W16 + j * K;
            const __fp16* w_row1 = W16 + (j + 1) * K;

            #if defined(__APPLE__) && defined(__aarch64__)
            if (j + 2 < end) {
                __builtin_prefetch(W16 + (j + 2) * K, 0, 2);
                __builtin_prefetch(W16 + (j + 3) * K, 0, 2);
            }
            #endif

            float16x8_t v00 = vdupq_n_f16(0.0f), v01 = vdupq_n_f16(0.0f);
            float16x8_t v02 = vdupq_n_f16(0.0f), v03 = vdupq_n_f16(0.0f);
            float16x8_t v10 = vdupq_n_f16(0.0f), v11 = vdupq_n_f16(0.0f);
            float16x8_t v12 = vdupq_n_f16(0.0f), v13 = vdupq_n_f16(0.0f);
            int64_t k = 0;
            for (; k <= K - 32; k += 32) {
                #if defined(__APPLE__) && defined(__aarch64__)
                __builtin_prefetch(w_row0 + k + 64, 0, 3);
                __builtin_prefetch(w_row1 + k + 64, 0, 3);
                #endif
                float16x8_t x0 = vld1q_f16(x16 + k);
                float16x8_t x1 = vld1q_f16(x16 + k + 8);
                float16x8_t x2 = vld1q_f16(x16 + k + 16);
                float16x8_t x3 = vld1q_f16(x16 + k + 24);

                v00 = vfmaq_f16(v00, x0, vld1q_f16(w_row0 + k));
                v10 = vfmaq_f16(v10, x0, vld1q_f16(w_row1 + k));

                v01 = vfmaq_f16(v01, x1, vld1q_f16(w_row0 + k + 8));
                v11 = vfmaq_f16(v11, x1, vld1q_f16(w_row1 + k + 8));

                v02 = vfmaq_f16(v02, x2, vld1q_f16(w_row0 + k + 16));
                v12 = vfmaq_f16(v12, x2, vld1q_f16(w_row1 + k + 16));

                v03 = vfmaq_f16(v03, x3, vld1q_f16(w_row0 + k + 24));
                v13 = vfmaq_f16(v13, x3, vld1q_f16(w_row1 + k + 24));
            }
            float16x8_t vsum0 = vaddq_f16(vaddq_f16(v00, v01), vaddq_f16(v02, v03));
            float16x8_t vsum1 = vaddq_f16(vaddq_f16(v10, v11), vaddq_f16(v12, v13));

            float16x4_t vsum_low0 = vadd_f16(vget_low_f16(vsum0), vget_high_f16(vsum0));
            float16x4_t vsum_low1 = vadd_f16(vget_low_f16(vsum1), vget_high_f16(vsum1));

            float sum0 = vaddvq_f32(vcvt_f32_f16(vsum_low0));
            float sum1 = vaddvq_f32(vcvt_f32_f16(vsum_low1));

            for (; k < K; ++k) {
                sum0 += static_cast<float>(x16[k]) * static_cast<float>(w_row0[k]);
                sum1 += static_cast<float>(x16[k]) * static_cast<float>(w_row1[k]);
            }
            if (acc) {
                out[j] += sum0;
                out[j + 1] += sum1;
            } else {
                out[j] = sum0;
                out[j + 1] = sum1;
            }
        }
        for (; j < end; ++j) {
            const __fp16* w_row = W16 + j * K;
            float16x8_t v0 = vdupq_n_f16(0.0f), v1 = vdupq_n_f16(0.0f);
            float16x8_t v2 = vdupq_n_f16(0.0f), v3 = vdupq_n_f16(0.0f);
            int64_t k = 0;
            for (; k <= K - 32; k += 32) {
                v0 = vfmaq_f16(v0, vld1q_f16(x16 + k), vld1q_f16(w_row + k));
                v1 = vfmaq_f16(v1, vld1q_f16(x16 + k + 8), vld1q_f16(w_row + k + 8));
                v2 = vfmaq_f16(v2, vld1q_f16(x16 + k + 16), vld1q_f16(w_row + k + 16));
                v3 = vfmaq_f16(v3, vld1q_f16(x16 + k + 24), vld1q_f16(w_row + k + 24));
            }
            float16x8_t vsum = vaddq_f16(vaddq_f16(v0, v1), vaddq_f16(v2, v3));
            float16x4_t vsum_low = vadd_f16(vget_low_f16(vsum), vget_high_f16(vsum));
            float sum = vaddvq_f32(vcvt_f32_f16(vsum_low));
            for (; k < K; ++k) sum += static_cast<float>(x16[k]) * static_cast<float>(w_row[k]);
            if (acc) out[j] += sum;
            else out[j] = sum;
        }
    } else if (task == TASK_SWIGLU) {
        int64_t intermediate = curr_N_;
        if (!sparse_mode_) {
            // Full Dense Mode (Default: 100% parameter compute with prefetching)
            for (int64_t j = start; j < end; ++j) {
                const __fp16* w_gate = W16 + j * K;
                const __fp16* w_up = W16 + (j + intermediate) * K;

                #if defined(__APPLE__) && defined(__aarch64__)
                if (j + 1 < end) {
                    __builtin_prefetch(W16 + (j + 1) * K, 0, 2);
                    __builtin_prefetch(W16 + (j + 1 + intermediate) * K, 0, 2);
                }
                #endif

                float16x8_t vg0 = vdupq_n_f16(0.0f), vg1 = vdupq_n_f16(0.0f);
                float16x8_t vg2 = vdupq_n_f16(0.0f), vg3 = vdupq_n_f16(0.0f);
                float16x8_t vu0 = vdupq_n_f16(0.0f), vu1 = vdupq_n_f16(0.0f);
                float16x8_t vu2 = vdupq_n_f16(0.0f), vu3 = vdupq_n_f16(0.0f);
                int64_t k = 0;
                for (; k <= K - 32; k += 32) {
                    #if defined(__APPLE__) && defined(__aarch64__)
                    __builtin_prefetch(w_gate + k + 64, 0, 3);
                    __builtin_prefetch(w_up + k + 64, 0, 3);
                    #endif
                    float16x8_t x0 = vld1q_f16(x16 + k);
                    float16x8_t x1 = vld1q_f16(x16 + k + 8);
                    float16x8_t x2 = vld1q_f16(x16 + k + 16);
                    float16x8_t x3 = vld1q_f16(x16 + k + 24);

                    vg0 = vfmaq_f16(vg0, x0, vld1q_f16(w_gate + k));
                    vg1 = vfmaq_f16(vg1, x1, vld1q_f16(w_gate + k + 8));
                    vg2 = vfmaq_f16(vg2, x2, vld1q_f16(w_gate + k + 16));
                    vg3 = vfmaq_f16(vg3, x3, vld1q_f16(w_gate + k + 24));

                    vu0 = vfmaq_f16(vu0, x0, vld1q_f16(w_up + k));
                    vu1 = vfmaq_f16(vu1, x1, vld1q_f16(w_up + k + 8));
                    vu2 = vfmaq_f16(vu2, x2, vld1q_f16(w_up + k + 16));
                    vu3 = vfmaq_f16(vu3, x3, vld1q_f16(w_up + k + 24));
                }
                float16x8_t vg_sum = vaddq_f16(vaddq_f16(vg0, vg1), vaddq_f16(vg2, vg3));
                float16x8_t vu_sum = vaddq_f16(vaddq_f16(vu0, vu1), vaddq_f16(vu2, vu3));

                float16x4_t vg_low = vadd_f16(vget_low_f16(vg_sum), vget_high_f16(vg_sum));
                float16x4_t vu_low = vadd_f16(vget_low_f16(vu_sum), vget_high_f16(vu_sum));

                float gate = vaddvq_f32(vcvt_f32_f16(vg_low));
                float up = vaddvq_f32(vcvt_f32_f16(vu_low));
                for (; k < K; ++k) {
                    gate += static_cast<float>(x16[k]) * static_cast<float>(w_gate[k]);
                    up += static_cast<float>(x16[k]) * static_cast<float>(w_up[k]);
                }
                float silu = gate / (1.0f + expf(-gate));
                out[j] = silu * up;
            }
        } else {
            // Dynamic Activation Sparsity (PowerInfer-style conditional execution)
            for (int64_t j = start; j < end; ++j) {
                const __fp16* w_gate = W16 + j * K;
                #if defined(__APPLE__) && defined(__aarch64__)
                if (j + 1 < end) __builtin_prefetch(W16 + (j + 1) * K, 0, 2);
                #endif

                // 1. Compute gate dot product first
                float16x8_t vg0 = vdupq_n_f16(0.0f), vg1 = vdupq_n_f16(0.0f);
                float16x8_t vg2 = vdupq_n_f16(0.0f), vg3 = vdupq_n_f16(0.0f);
                int64_t k = 0;
                for (; k <= K - 32; k += 32) {
                    #if defined(__APPLE__) && defined(__aarch64__)
                    __builtin_prefetch(w_gate + k + 64, 0, 3);
                    #endif
                    float16x8_t x0 = vld1q_f16(x16 + k);
                    float16x8_t x1 = vld1q_f16(x16 + k + 8);
                    float16x8_t x2 = vld1q_f16(x16 + k + 16);
                    float16x8_t x3 = vld1q_f16(x16 + k + 24);

                    vg0 = vfmaq_f16(vg0, x0, vld1q_f16(w_gate + k));
                    vg1 = vfmaq_f16(vg1, x1, vld1q_f16(w_gate + k + 8));
                    vg2 = vfmaq_f16(vg2, x2, vld1q_f16(w_gate + k + 16));
                    vg3 = vfmaq_f16(vg3, x3, vld1q_f16(w_gate + k + 24));
                }
                float16x8_t vg_sum = vaddq_f16(vaddq_f16(vg0, vg1), vaddq_f16(vg2, vg3));
                float16x4_t vg_low = vadd_f16(vget_low_f16(vg_sum), vget_high_f16(vg_sum));
                float gate = vaddvq_f32(vcvt_f32_f16(vg_low));
                for (; k < K; ++k) gate += static_cast<float>(x16[k]) * static_cast<float>(w_gate[k]);

                // 2. Threshold check: if gate < threshold, SiLU is near 0.0 -> skip w_up entirely!
                if (gate < sparse_threshold_) {
                    out[j] = 0.0f;
                    continue;
                }

                // 3. Only active neurons compute w_up
                const __fp16* w_up = W16 + (j + intermediate) * K;
                float16x8_t vu0 = vdupq_n_f16(0.0f), vu1 = vdupq_n_f16(0.0f);
                float16x8_t vu2 = vdupq_n_f16(0.0f), vu3 = vdupq_n_f16(0.0f);
                k = 0;
                for (; k <= K - 32; k += 32) {
                    #if defined(__APPLE__) && defined(__aarch64__)
                    __builtin_prefetch(w_up + k + 64, 0, 3);
                    #endif
                    float16x8_t x0 = vld1q_f16(x16 + k);
                    float16x8_t x1 = vld1q_f16(x16 + k + 8);
                    float16x8_t x2 = vld1q_f16(x16 + k + 16);
                    float16x8_t x3 = vld1q_f16(x16 + k + 24);

                    vu0 = vfmaq_f16(vu0, x0, vld1q_f16(w_up + k));
                    vu1 = vfmaq_f16(vu1, x1, vld1q_f16(w_up + k + 8));
                    vu2 = vfmaq_f16(vu2, x2, vld1q_f16(w_up + k + 16));
                    vu3 = vfmaq_f16(vu3, x3, vld1q_f16(w_up + k + 24));
                }
                float16x8_t vu_sum = vaddq_f16(vaddq_f16(vu0, vu1), vaddq_f16(vu2, vu3));
                float16x4_t vu_low = vadd_f16(vget_low_f16(vu_sum), vget_high_f16(vu_sum));
                float up = vaddvq_f32(vcvt_f32_f16(vu_low));
                for (; k < K; ++k) up += static_cast<float>(x16[k]) * static_cast<float>(w_up[k]);

                float silu = gate / (1.0f + expf(-gate));
                out[j] = silu * up;
            }
        }
    } else if (task == TASK_LM_HEAD) {
        float local_max = -1e30f;
        int64_t local_best = start;
        int64_t j = start;
        for (; j <= end - 2; j += 2) {
            const __fp16* w_row0 = W16 + j * K;
            const __fp16* w_row1 = W16 + (j + 1) * K;

            #if defined(__APPLE__) && defined(__aarch64__)
            if (j + 2 < end) {
                __builtin_prefetch(W16 + (j + 2) * K, 0, 2);
                __builtin_prefetch(W16 + (j + 3) * K, 0, 2);
            }
            #endif

            float16x8_t v00 = vdupq_n_f16(0.0f), v01 = vdupq_n_f16(0.0f);
            float16x8_t v02 = vdupq_n_f16(0.0f), v03 = vdupq_n_f16(0.0f);
            float16x8_t v10 = vdupq_n_f16(0.0f), v11 = vdupq_n_f16(0.0f);
            float16x8_t v12 = vdupq_n_f16(0.0f), v13 = vdupq_n_f16(0.0f);
            int64_t k = 0;
            for (; k <= K - 32; k += 32) {
                #if defined(__APPLE__) && defined(__aarch64__)
                __builtin_prefetch(w_row0 + k + 64, 0, 3);
                __builtin_prefetch(w_row1 + k + 64, 0, 3);
                #endif
                float16x8_t x0 = vld1q_f16(x16 + k);
                float16x8_t x1 = vld1q_f16(x16 + k + 8);
                float16x8_t x2 = vld1q_f16(x16 + k + 16);
                float16x8_t x3 = vld1q_f16(x16 + k + 24);

                v00 = vfmaq_f16(v00, x0, vld1q_f16(w_row0 + k));
                v10 = vfmaq_f16(v10, x0, vld1q_f16(w_row1 + k));

                v01 = vfmaq_f16(v01, x1, vld1q_f16(w_row0 + k + 8));
                v11 = vfmaq_f16(v11, x1, vld1q_f16(w_row1 + k + 8));

                v02 = vfmaq_f16(v02, x2, vld1q_f16(w_row0 + k + 16));
                v12 = vfmaq_f16(v12, x2, vld1q_f16(w_row1 + k + 16));

                v03 = vfmaq_f16(v03, x3, vld1q_f16(w_row0 + k + 24));
                v13 = vfmaq_f16(v13, x3, vld1q_f16(w_row1 + k + 24));
            }
            float16x8_t vsum0 = vaddq_f16(vaddq_f16(v00, v01), vaddq_f16(v02, v03));
            float16x8_t vsum1 = vaddq_f16(vaddq_f16(v10, v11), vaddq_f16(v12, v13));

            float16x4_t vsum_low0 = vadd_f16(vget_low_f16(vsum0), vget_high_f16(vsum0));
            float16x4_t vsum_low1 = vadd_f16(vget_low_f16(vsum1), vget_high_f16(vsum1));

            float sum0 = vaddvq_f32(vcvt_f32_f16(vsum_low0));
            float sum1 = vaddvq_f32(vcvt_f32_f16(vsum_low1));

            for (; k < K; ++k) {
                sum0 += static_cast<float>(x16[k]) * static_cast<float>(w_row0[k]);
                sum1 += static_cast<float>(x16[k]) * static_cast<float>(w_row1[k]);
            }
            out[j] = sum0;
            out[j + 1] = sum1;
            if (sum0 > local_max) {
                local_max = sum0;
                local_best = j;
            }
            if (sum1 > local_max) {
                local_max = sum1;
                local_best = j + 1;
            }
        }
        for (; j < end; ++j) {
            const __fp16* w_row = W16 + j * K;
            float16x8_t v0 = vdupq_n_f16(0.0f), v1 = vdupq_n_f16(0.0f);
            float16x8_t v2 = vdupq_n_f16(0.0f), v3 = vdupq_n_f16(0.0f);
            int64_t k = 0;
            for (; k <= K - 32; k += 32) {
                v0 = vfmaq_f16(v0, vld1q_f16(x16 + k), vld1q_f16(w_row + k));
                v1 = vfmaq_f16(v1, vld1q_f16(x16 + k + 8), vld1q_f16(w_row + k + 8));
                v2 = vfmaq_f16(v2, vld1q_f16(x16 + k + 16), vld1q_f16(w_row + k + 16));
                v3 = vfmaq_f16(v3, vld1q_f16(x16 + k + 24), vld1q_f16(w_row + k + 24));
            }
            float16x8_t vsum = vaddq_f16(vaddq_f16(v0, v1), vaddq_f16(v2, v3));
            float16x4_t vsum_low = vadd_f16(vget_low_f16(vsum), vget_high_f16(vsum));
            float sum = vaddvq_f32(vcvt_f32_f16(vsum_low));
            for (; k < K; ++k) sum += static_cast<float>(x16[k]) * static_cast<float>(w_row[k]);
            out[j] = sum;
            if (sum > local_max) {
                local_max = sum;
                local_best = j;
            }
        }
        local_max_[tid] = local_max;
        local_best_[tid] = local_best;
    }
}

void FastLlamaDecoder::run_parallel_task(DecoderTaskType task, const void* weight, float* out, int64_t K, int64_t N) {
    curr_task_.store(task, std::memory_order_relaxed);
    curr_weight_ = weight;
    curr_out_ = out;
    curr_K_ = K;
    curr_N_ = N;

    barrier_start_.wait(0);
    execute_worker_slice(0);
    barrier_end_.wait(0);
}

int64_t FastLlamaDecoder::decode_step(int64_t token_id, int64_t start_pos) {
    if (token_id < 0 || token_id >= vocab_size_) {
        throw std::runtime_error("token_id out of range");
    }
    if (start_pos >= max_seq_len_) {
        throw std::runtime_error("start_pos exceeds max_seq_len");
    }

    // 1. Embedding lookup
#if defined(__APPLE__) && defined(__aarch64__)
    const __fp16* w_embed = reinterpret_cast<const __fp16*>(embed_tokens_weight_) + token_id * dim_;
    int64_t i = 0;
    for (; i <= dim_ - 8; i += 8) {
        float16x8_t h16 = vld1q_f16(w_embed + i);
        vst1q_f32(h_buf_.data() + i, vcvt_f32_f16(vget_low_f16(h16)));
        vst1q_f32(h_buf_.data() + i + 4, vcvt_f32_f16(vget_high_f16(h16)));
    }
    for (; i < dim_; ++i) {
        h_buf_[i] = static_cast<float>(w_embed[i]);
    }
#else
    const float* w_embed = reinterpret_cast<const float*>(embed_tokens_weight_) + token_id * dim_;
    std::memcpy(h_buf_.data(), w_embed, dim_ * sizeof(float));
#endif

    const float* cos_ptr = cos_table_.data() + start_pos * head_dim_;
    const float* sin_ptr = sin_table_.data() + start_pos * head_dim_;
    int64_t qkv_dim = (num_heads_ + 2 * num_kv_heads_) * head_dim_;

    // 2. Transformer layers
    for (int64_t l = 0; l < num_layers_; ++l) {
        const auto& layer = layers_[l];

        // RMSNorm 1
        kernel_rmsnorm_f32(h_buf_.data(), layer.input_layernorm_weight, norm_buf_.data(), 1, dim_, eps_);
        float_to_fp16_neon(norm_buf_.data(), reinterpret_cast<__fp16*>(x16_buf_.data()), dim_);

        // Parallel QKV Projection
        run_parallel_task(TASK_QKV, layer.qkv_weight, qkv_buf_.data(), dim_, qkv_dim);

        if (layer.qkv_bias) {
            for (int64_t i = 0; i < qkv_dim; ++i) {
                qkv_buf_[i] += layer.qkv_bias[i];
            }
        }

        // Fused Decode Attention (with RoPE & KV Cache update)
        kernel_attention_decode_f32(
            qkv_buf_.data(),
            qkv_buf_.data() + num_heads_ * head_dim_,
            qkv_buf_.data() + (num_heads_ + num_kv_heads_) * head_dim_,
            layer.k_cache,
            layer.v_cache,
            cos_ptr,
            sin_ptr,
            attn_out_buf_.data(),
            num_heads_,
            num_kv_heads_,
            head_dim_,
            max_seq_len_,
            start_pos
        );
        float_to_fp16_neon(attn_out_buf_.data(), reinterpret_cast<__fp16*>(x16_buf_.data()), num_heads_ * head_dim_);

        // Parallel O Projection (fused with Residual 1 addition into h_buf_)
        run_parallel_task(TASK_O, layer.o_weight, h_buf_.data(), num_heads_ * head_dim_, dim_);

        // RMSNorm 2
        kernel_rmsnorm_f32(h_buf_.data(), layer.post_attn_layernorm_weight, norm2_buf_.data(), 1, dim_, eps_);
        float_to_fp16_neon(norm2_buf_.data(), reinterpret_cast<__fp16*>(x16_buf_.data()), dim_);

        // Parallel Fused Gate-Up GEMV + SwiGLU
        run_parallel_task(TASK_SWIGLU, layer.gate_up_weight, swiglu_buf_.data(), dim_, intermediate_size_);
        float_to_fp16_neon(swiglu_buf_.data(), reinterpret_cast<__fp16*>(x16_buf_.data()), intermediate_size_);

        // Parallel Down Projection (fused with Residual 2 addition into h_buf_)
        run_parallel_task(TASK_DOWN, layer.down_weight, h_buf_.data(), intermediate_size_, dim_);
    }

    // 3. Final RMSNorm
    kernel_rmsnorm_f32(h_buf_.data(), norm_weight_, norm_buf_.data(), 1, dim_, eps_);
    float_to_fp16_neon(norm_buf_.data(), reinterpret_cast<__fp16*>(x16_buf_.data()), dim_);

    // 4. Parallel LM Head Projection with distributed argmax
    run_parallel_task(TASK_LM_HEAD, lm_head_weight_, logits_buf_.data(), dim_, vocab_size_);

    // 5. Global Argmax across 4 threads
    float best_val = local_max_[0];
    int64_t best_tok = local_best_[0];
    for (int t = 1; t < num_threads_; ++t) {
        if (local_max_[t] > best_val) {
            best_val = local_max_[t];
            best_tok = local_best_[t];
        }
    }

    return best_tok;
}

Tensor FastLlamaDecoder::decode_step_logits(int64_t token_id, int64_t start_pos) {
    decode_step(token_id, start_pos);
    Tensor out = Tensor::empty({1, vocab_size_}, DType::Float32, Device::CPU());
    std::memcpy(out.data_ptr<float>(), logits_buf_.data(), vocab_size_ * sizeof(float));
    return out;
}

Tensor FastLlamaDecoder::prefill_logits(const std::vector<int64_t>& prompt_tokens) {
    prefill(prompt_tokens);
    Tensor out = Tensor::empty({1, vocab_size_}, DType::Float32, Device::CPU());
    std::memcpy(out.data_ptr<float>(), logits_buf_.data(), vocab_size_ * sizeof(float));
    return out;
}

int64_t FastLlamaDecoder::prefill(const std::vector<int64_t>& prompt_tokens) {
    if (prompt_tokens.empty()) {
        throw std::runtime_error("prompt_tokens cannot be empty");
    }
    int64_t N = prompt_tokens.size();
    if (N > max_seq_len_) {
        throw std::runtime_error("prompt length exceeds max_seq_len");
    }
    if (N == 1) {
        return decode_step(prompt_tokens[0], 0);
    }

    int64_t qkv_dim = (num_heads_ + 2 * num_kv_heads_) * head_dim_;

    // 1. Batch embedding lookup for all N prompt tokens
    std::vector<float> h_batch(N * dim_);
    for (int64_t pos = 0; pos < N; ++pos) {
        int64_t token_id = prompt_tokens[pos];
#if defined(__APPLE__) && defined(__aarch64__)
        const __fp16* w_embed = reinterpret_cast<const __fp16*>(embed_tokens_weight_) + token_id * dim_;
        float* h_p = h_batch.data() + pos * dim_;
        int64_t i = 0;
        for (; i <= dim_ - 8; i += 8) {
            float16x8_t h16 = vld1q_f16(w_embed + i);
            vst1q_f32(h_p + i, vcvt_f32_f16(vget_low_f16(h16)));
            vst1q_f32(h_p + i + 4, vcvt_f32_f16(vget_high_f16(h16)));
        }
        for (; i < dim_; ++i) h_p[i] = static_cast<float>(w_embed[i]);
#else
        const float* w_embed = reinterpret_cast<const float*>(embed_tokens_weight_) + token_id * dim_;
        std::memcpy(h_batch.data() + pos * dim_, w_embed, dim_ * sizeof(float));
#endif
    }

    // 2. Layer-by-layer execution: weights loaded ONCE into cache per layer for all tokens
    for (int64_t l = 0; l < num_layers_; ++l) {
        const auto& layer = layers_[l];

        for (int64_t pos = 0; pos < N; ++pos) {
            float* h_p = h_batch.data() + pos * dim_;

            // RMSNorm 1
            kernel_rmsnorm_f32(h_p, layer.input_layernorm_weight, norm_buf_.data(), 1, dim_, eps_);
            float_to_fp16_neon(norm_buf_.data(), reinterpret_cast<__fp16*>(x16_buf_.data()), dim_);

            // QKV GEMV
            run_parallel_task(TASK_QKV, layer.qkv_weight, qkv_buf_.data(), dim_, qkv_dim);

            if (layer.qkv_bias) {
                for (int64_t i = 0; i < qkv_dim; ++i) {
                    qkv_buf_[i] += layer.qkv_bias[i];
                }
            }

            // Attention (applies RoPE at position 'pos', updates KV cache, and computes context)
            const float* cos_ptr = cos_table_.data() + pos * head_dim_;
            const float* sin_ptr = sin_table_.data() + pos * head_dim_;
            kernel_attention_decode_f32(
                qkv_buf_.data(),
                qkv_buf_.data() + num_heads_ * head_dim_,
                qkv_buf_.data() + (num_heads_ + num_kv_heads_) * head_dim_,
                layer.k_cache,
                layer.v_cache,
                cos_ptr,
                sin_ptr,
                attn_out_buf_.data(),
                num_heads_,
                num_kv_heads_,
                head_dim_,
                max_seq_len_,
                pos
            );
            float_to_fp16_neon(attn_out_buf_.data(), reinterpret_cast<__fp16*>(x16_buf_.data()), num_heads_ * head_dim_);

            // O GEMV (accumulates into h_p)
            run_parallel_task(TASK_O, layer.o_weight, h_p, num_heads_ * head_dim_, dim_);

            // RMSNorm 2
            kernel_rmsnorm_f32(h_p, layer.post_attn_layernorm_weight, norm2_buf_.data(), 1, dim_, eps_);
            float_to_fp16_neon(norm2_buf_.data(), reinterpret_cast<__fp16*>(x16_buf_.data()), dim_);

            // SwiGLU GEMV
            run_parallel_task(TASK_SWIGLU, layer.gate_up_weight, swiglu_buf_.data(), dim_, intermediate_size_);
            float_to_fp16_neon(swiglu_buf_.data(), reinterpret_cast<__fp16*>(x16_buf_.data()), intermediate_size_);

            // Down GEMV (accumulates into h_p)
            run_parallel_task(TASK_DOWN, layer.down_weight, h_p, intermediate_size_, dim_);
        }
    }

    // 3. Final token (N-1): compute final RMSNorm + LM Head to produce first generated token
    float* h_last = h_batch.data() + (N - 1) * dim_;
    std::memcpy(h_buf_.data(), h_last, dim_ * sizeof(float));

    kernel_rmsnorm_f32(h_buf_.data(), norm_weight_, norm_buf_.data(), 1, dim_, eps_);
    float_to_fp16_neon(norm_buf_.data(), reinterpret_cast<__fp16*>(x16_buf_.data()), dim_);

    run_parallel_task(TASK_LM_HEAD, lm_head_weight_, logits_buf_.data(), dim_, vocab_size_);

    float best_val = local_max_[0];
    int64_t best_tok = local_best_[0];
    for (int t = 1; t < num_threads_; ++t) {
        if (local_max_[t] > best_val) {
            best_val = local_max_[t];
            best_tok = local_best_[t];
        }
    }

    return best_tok;
}

} // namespace velocityai

