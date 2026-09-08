#include "csrc/kernels/cpu/simd_utils.h"
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Online softmax or simple 3-pass softmax
void kernel_softmax_f32(const float* in, float* out, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* row_in = in + r * cols;
        float* row_out = out + r * cols;
        
        // 1. Find max
        float max_val = row_in[0];
        for (int64_t c = 1; c < cols; ++c) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }
        
        // 2. Exp and sum
        float sum_exp = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            row_out[c] = expf(row_in[c] - max_val);
            sum_exp += row_out[c];
        }
        
        // 3. Normalize
        float inv_sum = 1.0f / sum_exp;
        int64_t c = 0;
#ifdef VAI_USE_NEON
        float32x4_t vinv = vdupq_n_f32(inv_sum);
        for (; c <= cols - 4; c += 4) {
            float32x4_t vout = vld1q_f32(row_out + c);
            vout = vmulq_f32(vout, vinv);
            vst1q_f32(row_out + c, vout);
        }
#endif
        for (; c < cols; ++c) {
            row_out[c] *= inv_sum;
        }
    }
}

#ifdef __cplusplus
}
#endif
