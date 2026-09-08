#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Embedding lookup
void kernel_embedding_f32(const int64_t* indices, const float* weight, float* out, 
                          int64_t num_indices, int64_t embedding_dim) {
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = indices[i];
        const float* w_row = weight + idx * embedding_dim;
        float* out_row = out + i * embedding_dim;
        for (int64_t d = 0; d < embedding_dim; ++d) {
            out_row[d] = w_row[d];
        }
    }
}

#ifdef __cplusplus
}
#endif
