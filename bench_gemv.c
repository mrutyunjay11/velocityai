
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define K 960
#define N 2560
#define ITERS 10000

__attribute__((noinline)) void gemv_fp32(const float* x, const float* W, float* y) {
    for (int col = 0; col < N; ++col) {
        float sum = 0.0f;
        for (int row = 0; row < K; ++row) {
            sum += x[row] * W[row * N + col];
        }
        y[col] = sum;
    }
}

__attribute__((noinline)) void gemv_fp16(const float* x, const __fp16* W, float* y) {
    for (int col = 0; col < N; ++col) {
        float sum = 0.0f;
        for (int row = 0; row < K; ++row) {
            sum += x[row] * (float)W[row * N + col];
        }
        y[col] = sum;
    }
}

int main() {
    float* x = (float*)malloc(K * sizeof(float));
    float* W32 = (float*)malloc(K * N * sizeof(float));
    __fp16* W16 = (__fp16*)malloc(K * N * sizeof(__fp16));
    float* y = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < K; ++i) x[i] = 0.1f;
    for (int i = 0; i < K * N; ++i) { W32[i] = 0.01f; W16[i] = (__fp16)0.01f; }

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int it = 0; it < ITERS; ++it) gemv_fp32(x, W32, y);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double t32 = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int it = 0; it < ITERS; ++it) gemv_fp16(x, W16, y);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double t16 = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;

    printf("FP32 Time: %.4fs (%.2f GB/s)\n", t32, (double)ITERS * K * N * 4 / t32 / 1e9);
    printf("FP16 Time: %.4fs (%.2f GB/s)\n", t16, (double)ITERS * K * N * 2 / t16 / 1e9);
    printf("Speedup: %.2fx\n", t32 / t16);
    return 0;
}
