#include "csrc/kernels/metal/metal_backend.h"
#include <iostream>
#include <stdexcept>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <unordered_map>
#include <mutex>

namespace velocityai {
namespace metal {

static id<MTLDevice> device = nil;
static id<MTLCommandQueue> command_queue = nil;
static id<MTLComputePipelineState> matmul_pipeline = nil;

static auto* buffer_registry = new std::unordered_map<void*, id<MTLBuffer>>();
static auto* registry_mutex = new std::mutex();

id<MTLBuffer> get_metal_buffer(const void* ptr, size_t bytes) {
    std::lock_guard<std::mutex> lock(*registry_mutex);
    auto it = buffer_registry->find(const_cast<void*>(ptr));
    if (it != buffer_registry->end()) {
        return it->second;
    }
    // If not in registry (e.g., mmap'd or CPU malloc'd), create ephemeral buffer without copy
    // Note: newBufferWithBytesNoCopy requires page alignment, which we don't guarantee.
    // So for external CPU pointers, we fallback to newBufferWithBytes (which copies).
    return [device newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
}

const char* matmul_shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) 
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row < M && col < N) {
        float sum = 0.0;
        for (uint i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
)";

bool is_available() {
    if (!device) {
        device = MTLCreateSystemDefaultDevice();
    }
    return device != nil;
}

void init_metal() {
    if (!is_available()) return;
    
    command_queue = [device newCommandQueue];
    
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:matmul_shader];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    
    if (error) {
        std::cerr << "Metal shader compilation failed: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }
    
    id<MTLFunction> function = [library newFunctionWithName:@"matmul_kernel"];
    matmul_pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    
    if (error) {
        std::cerr << "Metal pipeline creation failed: " << [[error localizedDescription] UTF8String] << std::endl;
    }
}

void* allocate(size_t bytes) {
    if (!device) init_metal();
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    void* ptr = [buffer contents];
    
    std::lock_guard<std::mutex> lock(*registry_mutex);
    (*buffer_registry)[ptr] = buffer;
    
    return ptr;
}

void deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(*registry_mutex);
    buffer_registry->erase(ptr);
}

void matmul_metal(const Tensor& a, const Tensor& b, Tensor& out) {
    if (!matmul_pipeline) init_metal();
    
    uint M = a.shape(0);
    uint K = a.shape(1);
    uint N = b.shape(1);
    
    id<MTLCommandBuffer> commandBuffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:matmul_pipeline];
    
    // Get buffers from registry (zero-copy if allocated on Metal, copies if external)
    id<MTLBuffer> bufA = get_metal_buffer(a.data_ptr(), a.nbytes());
    id<MTLBuffer> bufB = get_metal_buffer(b.data_ptr(), b.nbytes());
    id<MTLBuffer> bufC = get_metal_buffer(out.data_ptr(), out.nbytes());
    
    [encoder setBuffer:bufA offset:0 atIndex:0];
    [encoder setBuffer:bufB offset:0 atIndex:1];
    [encoder setBuffer:bufC offset:0 atIndex:2];
    [encoder setBytes:&M length:sizeof(uint) atIndex:3];
    [encoder setBytes:&K length:sizeof(uint) atIndex:4];
    [encoder setBytes:&N length:sizeof(uint) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake(N, M, 1);
    
    NSUInteger max_threads = matmul_pipeline.maxTotalThreadsPerThreadgroup;
    // Simplistic threadgroup sizing
    NSUInteger w = 16;
    NSUInteger h = 16;
    if (w * h > max_threads) {
        w = 8; h = 8;
    }
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy result back
    memcpy(out.data_ptr(), [bufC contents], out.nbytes());
}

} // namespace metal
} // namespace velocityai

#else

// Non-Apple fallback
namespace velocityai {
namespace metal {
bool is_available() { return false; }
void init_metal() {}
void matmul_metal(const Tensor& a, const Tensor& b, Tensor& out) {
    throw std::runtime_error("Metal is not available on this platform.");
}
void* allocate(size_t bytes) { return nullptr; }
void deallocate(void* ptr) {}
}
}
#endif
