#include "csrc/kernels/cpu/simd_utils.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void kernel_sum_f32(const float* in, float* out, int64_t numel) {
    double total = 0.0;
    int64_t i = 0;
#ifdef VAI_USE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i <= numel - 4; i += 4) {
        float32x4_t va = vld1q_f32(in + i);
        vsum = vaddq_f32(vsum, va);
    }
    float temp[4];
    vst1q_f32(temp, vsum);
    total += temp[0] + temp[1] + temp[2] + temp[3];
#endif
    for (; i < numel; ++i) {
        total += in[i];
    }
    *out = (float)total;
}

#ifdef __cplusplus
}
#endif
