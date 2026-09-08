#ifndef VAI_SIMD_UTILS_H
#define VAI_SIMD_UTILS_H

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define VAI_USE_NEON
#endif

// Fallbacks for other architectures can be added here (AVX/AVX2/AVX512)

#endif
