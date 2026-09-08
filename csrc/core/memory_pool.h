#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace velocityai {

class MemoryPool {
public:
    static MemoryPool& instance();

    // Allocate aligned memory (default 64-byte alignment for AVX/NEON)
    void* allocate(size_t bytes, size_t alignment = 64);

    // Free memory back to the pool
    void deallocate(void* ptr, size_t bytes);

    // Release all pooled memory back to the operating system
    void empty_cache();

    // Memory statistics
    size_t allocated_bytes() const { return allocated_bytes_.load(); }
    size_t cached_bytes() const { return cached_bytes_.load(); }
    size_t peak_bytes() const { return peak_bytes_.load(); }

private:
    MemoryPool() = default;
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    size_t get_bin_size(size_t bytes) const;

    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<void*>> bins_;
    
    std::atomic<size_t> allocated_bytes_{0};
    std::atomic<size_t> cached_bytes_{0};
    std::atomic<size_t> peak_bytes_{0};

    // Cap pool caching to avoid unbounded memory accumulation (e.g. 512 MB)
    static constexpr size_t kMaxCachedBytes = 512 * 1024 * 1024;
    // Maximum single block size to pool (e.g. 32 MB)
    static constexpr size_t kMaxBinSize = 32 * 1024 * 1024;
};

} // namespace velocityai
