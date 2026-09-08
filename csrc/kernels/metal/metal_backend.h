#pragma once
#include "csrc/core/tensor.h"

namespace velocityai {
namespace metal {

bool is_available();
void init_metal();
void matmul_metal(const Tensor& a, const Tensor& b, Tensor& out);

// Allocate and free memory on Metal device
void* allocate(size_t bytes);
void deallocate(void* ptr);

} // namespace metal
} // namespace velocityai
