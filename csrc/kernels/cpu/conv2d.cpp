#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// im2col + GEMM convolution stub
void kernel_conv2d_f32(const float* in, const float* weight, const float* bias, float* out,
                       int64_t batch, int64_t in_ch, int64_t in_h, int64_t in_w,
                       int64_t out_ch, int64_t k_h, int64_t k_w,
                       int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w) {
    // To be implemented via im2col -> matmul.
}

#ifdef __cplusplus
}
#endif
