#include "csrc/core/memory_pool.h"
#include "csrc/core/error.h"
#include <cstdlib>
#include <algorithm>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace velocityai {

MemoryPool& MemoryPool::instance() {
    static MemoryPool* pool = new MemoryPool();
    return *pool;
}

MemoryPool::~MemoryPool() {
    empty_cache();
}

size_t MemoryPool::get_bin_size(size_t bytes) const {
    if (bytes == 0) return 64;
    // Round up to nearest power of two or multiples of 4KB
    if (bytes <= 512) {
        size_t b = 64;
        while (b < bytes) b <<= 1;
        return b;
    }
    // For larger allocations, round up to next multiple of 4KB
    size_t kPage = 4096;
    return ((bytes + kPage - 1) / kPage) * kPage;
}

void* MemoryPool::allocate(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;

    size_t bin_size = get_bin_size(bytes);

    // If eligible for pooling, try fetching from bin
    if (bin_size <= kMaxBinSize) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = bins_.find(bin_size);
        if (it != bins_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            cached_bytes_ -= bin_size;
            allocated_bytes_ += bin_size;
            size_t current = allocated_bytes_.load();
            size_t peak = peak_bytes_.load();
            while (current > peak && !peak_bytes_.compare_exchange_weak(peak, current)) {}
            return ptr;
        }
    }

    // Allocate new memory with alignment
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bin_size, alignment);
    VAI_CHECK(ptr != nullptr, "Out of memory: _aligned_malloc failed");
#else
    int ret = posix_memalign(&ptr, alignment, bin_size);
    VAI_CHECK(ret == 0 && ptr != nullptr, "Out of memory: posix_memalign failed");
#endif

    allocated_bytes_ += bin_size;
    size_t current = allocated_bytes_.load();
    size_t peak = peak_bytes_.load();
    while (current > peak && !peak_bytes_.compare_exchange_weak(peak, current)) {}

    return ptr;
}

void MemoryPool::deallocate(void* ptr, size_t bytes) {
    if (!ptr || bytes == 0) return;

    size_t bin_size = get_bin_size(bytes);

    if (bin_size <= kMaxBinSize) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cached_bytes_ + bin_size <= kMaxCachedBytes) {
            bins_[bin_size].push_back(ptr);
            cached_bytes_ += bin_size;
            allocated_bytes_ -= bin_size;
            return;
        }
    }

    allocated_bytes_ -= bin_size;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void MemoryPool::empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : bins_) {
        for (void* ptr : pair.second) {
#if defined(_MSC_VER)
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
        pair.second.clear();
    }
    cached_bytes_ = 0;
}

} // namespace velocityai
