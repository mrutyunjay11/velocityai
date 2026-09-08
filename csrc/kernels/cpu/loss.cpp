#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void kernel_cross_entropy_f32(const float* logits, const int64_t* targets, float* out_loss, int64_t batch, int64_t classes) {
    float total_loss = 0.0f;
    for (int64_t b = 0; b < batch; ++b) {
        const float* row = logits + b * classes;
        int64_t target = targets[b];
        
        float max_val = row[0];
        for (int64_t c = 1; c < classes; ++c) {
            if (row[c] > max_val) max_val = row[c];
        }
        
        float sum_exp = 0.0f;
        for (int64_t c = 0; c < classes; ++c) {
            sum_exp += expf(row[c] - max_val);
        }
        
        float log_prob = (row[target] - max_val) - logf(sum_exp);
        total_loss -= log_prob;
    }
    *out_loss = total_loss / batch;
}

#ifdef __cplusplus
}
#endif
