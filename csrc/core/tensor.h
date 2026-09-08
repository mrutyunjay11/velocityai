#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <iostream>
#include "csrc/core/dtype.h"
#include "csrc/core/device.h"
#include "csrc/core/memory_pool.h"
#include "csrc/core/error.h"

namespace velocityai {

namespace metal {
void* allocate(size_t bytes);
void deallocate(void* ptr);
}

struct Storage {
    void* data{nullptr};
    size_t bytes{0};
    Device device{DeviceType::CPU, 0};
    bool owns_data{true};

    Storage(size_t nbytes, Device dev, bool allocate = true)
        : bytes(nbytes), device(dev), owns_data(allocate) {
        if (allocate && nbytes > 0) {
            if (dev.is_cpu()) {
                data = MemoryPool::instance().allocate(nbytes);
            } else if (dev.is_metal()) {
                data = metal::allocate(nbytes);
            } else {
                // CUDA allocation placeholder for later phases
                throw DeviceError("CUDA allocation not yet initialized");
            }
        }
    }

    Storage(void* raw_ptr, size_t nbytes, Device dev, bool take_ownership = false)
        : data(raw_ptr), bytes(nbytes), device(dev), owns_data(take_ownership) {}

    ~Storage() {
        if (owns_data && data && bytes > 0) {
            if (device.is_cpu()) {
                MemoryPool::instance().deallocate(data, bytes);
            } else if (device.is_metal()) {
                metal::deallocate(data);
            }
            data = nullptr;
        }
    }

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
};

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU());
    Tensor(std::shared_ptr<Storage> storage, int64_t offset,
           const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
           DType dtype, Device device);

    // Factory methods
    static Tensor empty(const std::vector<int64_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU());
    static Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU());
    static Tensor ones(const std::vector<int64_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU());
    static Tensor full(const std::vector<int64_t>& shape, float value, DType dtype = DType::Float32, Device device = Device::CPU());
    static Tensor randn(const std::vector<int64_t>& shape, float mean = 0.0f, float std = 1.0f, DType dtype = DType::Float32, Device device = Device::CPU());
    static Tensor from_blob(void* data, const std::vector<int64_t>& shape, DType dtype = DType::Float32, Device device = Device::CPU(), bool copy = false);

    // Metadata accessors
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t shape(int64_t dim) const;
    int64_t stride(int64_t dim) const;
    int64_t offset() const { return offset_; }
    int64_t numel() const { return numel_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t itemsize() const { return dtype_size(dtype_); }
    size_t nbytes() const { return numel_ * itemsize(); }
    bool is_contiguous() const;

    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool req) { requires_grad_ = req; }

    // Pointer access
    void* data_ptr();
    const void* data_ptr() const;

    template <typename T>
    T* data_ptr() {
        return reinterpret_cast<T*>(data_ptr());
    }

    template <typename T>
    const T* data_ptr() const {
        return reinterpret_cast<const T*>(data_ptr());
    }

    // View & layout manipulation (zero-copy when possible)
    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    Tensor view(const std::vector<int64_t>& new_shape) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor permute(const std::vector<int64_t>& dims) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;
    Tensor contiguous() const;
    Tensor clone() const;
    Tensor to(Device new_device) const;

    // Mutating in-place methods
    Tensor& zero_();
    Tensor& fill_(float val);

    // Basic arithmetic operations
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

    Tensor add_scalar(float val) const;
    Tensor mul_scalar(float val) const;

    // Reductions
    Tensor sum(int64_t dim = -1, bool keepdim = false) const;
    Tensor mean(int64_t dim = -1, bool keepdim = false) const;

    // Activations
    Tensor relu() const;

    // Operator overloads
    Tensor operator+(const Tensor& other) const { return add(other); }
    Tensor operator-(const Tensor& other) const { return sub(other); }
    Tensor operator*(const Tensor& other) const { return mul(other); }
    Tensor operator/(const Tensor& other) const { return div(other); }

    std::string to_string() const;

    static int64_t compute_numel(const std::vector<int64_t>& shape);

private:
    static std::vector<int64_t> compute_contiguous_strides(const std::vector<int64_t>& shape);

    std::shared_ptr<Storage> storage_;
    int64_t offset_{0}; // In elements
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_; // In elements
    int64_t numel_{0};
    DType dtype_{DType::Float32};
    Device device_{DeviceType::CPU, 0};
    bool requires_grad_{false};
};

} // namespace velocityai
