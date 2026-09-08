#include "csrc/kernels/cpu/simd_utils.h"
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Fused LayerNorm: mean, variance, normalize, scale, shift
void kernel_layernorm_f32(const float* in, const float* weight, const float* bias, 
                          float* out, int64_t rows, int64_t cols, float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* row_in = in + r * cols;
        float* row_out = out + r * cols;
        
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) sum += row_in[c];
        float mean = sum / cols;
        
        float var_sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float diff = row_in[c] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / cols;
        float inv_std = 1.0f / sqrtf(var + eps);
        
        int64_t c = 0;
#ifdef VAI_USE_NEON
        float32x4_t vmean = vdupq_n_f32(mean);
        float32x4_t vinv_std = vdupq_n_f32(inv_std);
        for (; c <= cols - 4; c += 4) {
            float32x4_t vin = vld1q_f32(row_in + c);
            float32x4_t vw = vld1q_f32(weight + c);
            float32x4_t vb = vld1q_f32(bias + c);
            
            float32x4_t norm = vmulq_f32(vsubq_f32(vin, vmean), vinv_std);
            float32x4_t vout = vaddq_f32(vmulq_f32(norm, vw), vb);
            vst1q_f32(row_out + c, vout);
        }
#endif
        for (; c < cols; ++c) {
            row_out[c] = (row_in[c] - mean) * inv_std * weight[c] + bias[c];
        }
    }
}

// Fused RMSNorm: out = (in / sqrt(mean(in^2) + eps)) * weight
void kernel_rmsnorm_f32(const float* in, const float* weight, float* out, int64_t rows, int64_t cols, float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* row_in = in + r * cols;
        float* row_out = out + r * cols;
        
        float sq_sum = 0.0f;
        int64_t c = 0;
#ifdef VAI_USE_NEON
        float32x4_t vsq = vdupq_n_f32(0.0f);
        for (; c <= cols - 4; c += 4) {
            float32x4_t vin = vld1q_f32(row_in + c);
            vsq = vfmaq_f32(vsq, vin, vin);
        }
        sq_sum = vaddvq_f32(vsq);
#endif
        for (; c < cols; ++c) {
            sq_sum += row_in[c] * row_in[c];
        }
        float rms = 1.0f / sqrtf(sq_sum / cols + eps);
        
        c = 0;
#ifdef VAI_USE_NEON
        float32x4_t vrms = vdupq_n_f32(rms);
        for (; c <= cols - 4; c += 4) {
            float32x4_t vin = vld1q_f32(row_in + c);
            float32x4_t vw = vld1q_f32(weight + c);
            float32x4_t vout = vmulq_f32(vmulq_f32(vin, vrms), vw);
            vst1q_f32(row_out + c, vout);
        }
#endif
        for (; c < cols; ++c) {
            row_out[c] = (row_in[c] * rms) * weight[c];
        }
    }
}

// Fused RMSNorm directly to FP16: converts to __fp16 in the same pass, avoiding intermediate buffers
void kernel_rmsnorm_to_fp16(const float* in, const float* weight, void* out_fp16, int64_t cols, float eps) {
    __fp16* dst = reinterpret_cast<__fp16*>(out_fp16);
    float sq_sum = 0.0f;
    int64_t c = 0;
#ifdef VAI_USE_NEON
    float32x4_t vsq = vdupq_n_f32(0.0f);
    for (; c <= cols - 4; c += 4) {
        float32x4_t vin = vld1q_f32(in + c);
        vsq = vfmaq_f32(vsq, vin, vin);
    }
    sq_sum = vaddvq_f32(vsq);
#endif
    for (; c < cols; ++c) {
        sq_sum += in[c] * in[c];
    }
    float rms = 1.0f / sqrtf(sq_sum / cols + eps);

    c = 0;
#if defined(__APPLE__) && defined(__aarch64__)
    float32x4_t vrms = vdupq_n_f32(rms);
    for (; c <= cols - 8; c += 8) {
        float32x4_t v0 = vmulq_f32(vmulq_f32(vld1q_f32(in + c), vrms), vld1q_f32(weight + c));
        float32x4_t v1 = vmulq_f32(vmulq_f32(vld1q_f32(in + c + 4), vrms), vld1q_f32(weight + c + 4));
        float16x4_t h0 = vcvt_f16_f32(v0);
        float16x4_t h1 = vcvt_f16_f32(v1);
        vst1q_f16(dst + c, vcombine_f16(h0, h1));
    }
#endif
    for (; c < cols; ++c) {
        dst[c] = static_cast<__fp16>((in[c] * rms) * weight[c]);
    }
}

#ifdef __cplusplus
}
#endif
