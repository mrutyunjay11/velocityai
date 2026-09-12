#ifndef VAI_LLAMA_DECODER_H
#define VAI_LLAMA_DECODER_H

#include "csrc/core/tensor.h"
#include <vector>
#include <cstdint>
#include <memory>
#include <atomic>
#include <pthread.h>

namespace velocityai {

class SpinBarrier {
public:
    SpinBarrier(int count) : count_(count), step_(0) {
        for (int i = 0; i < 8; ++i) arrived_[i].store(0, std::memory_order_relaxed);
    }

    void wait(int tid) {
        int current_step = step_.load(std::memory_order_relaxed);
        arrived_[tid].store(current_step + 1, std::memory_order_release);

        if (tid == 0) {
            for (int i = 1; i < count_; ++i) {
                while (arrived_[i].load(std::memory_order_acquire) <= current_step) {
                    #if defined(__APPLE__) && defined(__aarch64__)
                    __builtin_arm_isb(15);
                    #endif
                }
            }
            step_.store(current_step + 1, std::memory_order_release);
        } else {
            while (step_.load(std::memory_order_acquire) <= current_step) {
                #if defined(__APPLE__) && defined(__aarch64__)
                __builtin_arm_isb(15);
                #endif
            }
        }
    }

private:
    int count_;
    std::atomic<int> step_;
    std::atomic<int> arrived_[8];
};

struct FastLayerWeights {
    const float* input_layernorm_weight;
    const void* qkv_weight;
    const void* o_weight;
    const float* post_attn_layernorm_weight;
    const void* gate_up_weight;
    const void* down_weight;
    const float* qkv_bias;
    float* k_cache;
    float* v_cache;
};

enum DecoderTaskType {
    TASK_NONE,
    TASK_QKV,
    TASK_O,
    TASK_SWIGLU,
    TASK_DOWN,
    TASK_LM_HEAD,
    TASK_STOP
};

class FastLlamaDecoder {
public:
    FastLlamaDecoder(
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
    );

    ~FastLlamaDecoder();

    void add_layer(
        const Tensor& input_layernorm_weight,
        const Tensor& qkv_weight,
        const Tensor& o_weight,
        const Tensor& post_attn_layernorm_weight,
        const Tensor& gate_up_weight,
        const Tensor& down_weight,
        Tensor& k_cache,
        Tensor& v_cache,
        const Tensor* qkv_bias = nullptr
    );

    int64_t decode_step(int64_t token_id, int64_t start_pos);
    Tensor decode_step_logits(int64_t token_id, int64_t start_pos);
    int64_t prefill(const std::vector<int64_t>& prompt_tokens);
    Tensor prefill_logits(const std::vector<int64_t>& prompt_tokens);

    void set_sparse_mode(bool enabled, float threshold = -3.5f) {
        sparse_mode_ = enabled;
        sparse_threshold_ = threshold;
    }
    bool get_sparse_mode() const { return sparse_mode_; }
    float get_sparse_threshold() const { return sparse_threshold_; }

private:
    void run_parallel_task(DecoderTaskType task, const void* weight, float* out, int64_t K, int64_t N);
    static void* worker_thread_func(void* arg);
    void execute_worker_slice(int tid);

    int64_t vocab_size_;
    int64_t dim_;
    int64_t num_layers_;
    int64_t num_heads_;
    int64_t num_kv_heads_;
    int64_t head_dim_;
    int64_t intermediate_size_;
    int64_t max_seq_len_;
    float eps_;
    float rope_theta_;

    const void* embed_tokens_weight_;
    const float* norm_weight_;
    const void* lm_head_weight_;

    std::vector<FastLayerWeights> layers_;
    std::vector<Tensor> pinned_tensors_;

    // Precomputed RoPE tables: (max_seq_len, head_dim)
    std::vector<float> cos_table_;
    std::vector<float> sin_table_;

    // Preallocated activation scratch buffers
    std::vector<float> h_buf_;
    std::vector<float> norm_buf_;
    std::vector<float> qkv_buf_;
    std::vector<float> attn_out_buf_;
    std::vector<float> norm2_buf_;
    std::vector<float> swiglu_buf_;
    std::vector<float> logits_buf_;

    // FP16 converted activation buffer for high-throughput GEMV
    std::vector<uint16_t> x16_buf_;

    // Persistent 4-thread pool with atomic spin-barrier
    int num_threads_;
    SpinBarrier barrier_start_;
    SpinBarrier barrier_end_;
    std::vector<pthread_t> worker_handles_;
    std::atomic<DecoderTaskType> curr_task_;

    const void* curr_weight_;
    float* curr_out_;
    int64_t curr_K_;
    int64_t curr_N_;

    float local_max_[8];
    int64_t local_best_[8];

    bool sparse_mode_ = false;
    float sparse_threshold_ = -3.5f;
};

} // namespace velocityai

#endif // VAI_LLAMA_DECODER_H
