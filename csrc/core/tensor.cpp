#include "csrc/core/tensor.h"
#include "csrc/kernels/cpu/cpu_kernels.h"
#include "csrc/kernels/metal/metal_backend.h"
#include <cstring>
#include <random>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace velocityai {

std::vector<int64_t> Tensor::compute_contiguous_strides(const std::vector<int64_t>& shape) {
    if (shape.empty()) return {};
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= (shape[i] > 0 ? shape[i] : 1);
    }
    return strides;
}

int64_t Tensor::compute_numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1; // 0-d scalar tensor has 1 element
    int64_t n = 1;
    for (auto dim : shape) {
        n *= dim;
    }
    return n;
}

Tensor::Tensor()
    : offset_(0), numel_(0), dtype_(DType::Float32), device_(Device::CPU()) {}

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype, Device device)
    : offset_(0), shape_(shape), dtype_(dtype), device_(device) {
    strides_ = compute_contiguous_strides(shape_);
    numel_ = compute_numel(shape_);
    size_t bytes = numel_ * itemsize();
    storage_ = std::make_shared<Storage>(bytes, device_, true);
}

Tensor::Tensor(std::shared_ptr<Storage> storage, int64_t offset,
               const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
               DType dtype, Device device)
    : storage_(storage), offset_(offset), shape_(shape), strides_(strides),
      dtype_(dtype), device_(device) {
    numel_ = compute_numel(shape_);
}

Tensor Tensor::empty(const std::vector<int64_t>& shape, DType dtype, Device device) {
    return Tensor(shape, dtype, device);
}

Tensor Tensor::zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    t.zero_();
    return t;
}

Tensor Tensor::ones(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    t.fill_(1.0f);
    return t;
}

Tensor Tensor::full(const std::vector<int64_t>& shape, float value, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    t.fill_(value);
    return t;
}

Tensor Tensor::randn(const std::vector<int64_t>& shape, float mean, float std, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    VAI_CHECK(dtype == DType::Float32, "randn currently supports Float32");
    float* ptr = t.data_ptr<float>();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    for (int64_t i = 0; i < t.numel(); ++i) {
        ptr[i] = dist(gen);
    }
    return t;
}

Tensor Tensor::from_blob(void* data, const std::vector<int64_t>& shape, DType dtype, Device device, bool copy) {
    int64_t n = compute_numel(shape);
    size_t nbytes = n * dtype_size(dtype);
    if (copy) {
        Tensor t(shape, dtype, device);
        std::memcpy(t.data_ptr(), data, nbytes);
        return t;
    } else {
        auto storage = std::make_shared<Storage>(data, nbytes, device, false);
        return Tensor(storage, 0, shape, compute_contiguous_strides(shape), dtype, device);
    }
}

int64_t Tensor::shape(int64_t dim) const {
    if (dim < 0) dim += ndim();
    VAI_CHECK(dim >= 0 && dim < ndim(), "Dimension out of range");
    return shape_[dim];
}

int64_t Tensor::stride(int64_t dim) const {
    if (dim < 0) dim += ndim();
    VAI_CHECK(dim >= 0 && dim < ndim(), "Dimension out of range");
    return strides_[dim];
}

bool Tensor::is_contiguous() const {
    if (shape_.empty() || numel_ == 0) return true;
    int64_t expected_stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (shape_[i] == 1) continue;
        if (strides_[i] != expected_stride) return false;
        expected_stride *= shape_[i];
    }
    return true;
}

void* Tensor::data_ptr() {
    VAI_CHECK(storage_ && storage_->data, "Tensor has no allocated storage");
    return static_cast<char*>(storage_->data) + offset_ * itemsize();
}

const void* Tensor::data_ptr() const {
    VAI_CHECK(storage_ && storage_->data, "Tensor has no allocated storage");
    return static_cast<const char*>(storage_->data) + offset_ * itemsize();
}

Tensor& Tensor::zero_() {
    if (is_contiguous()) {
        std::memset(data_ptr(), 0, nbytes());
    } else {
        fill_(0.0f);
    }
    return *this;
}

Tensor& Tensor::fill_(float val) {
    VAI_CHECK(dtype_ == DType::Float32, "fill_ currently supports Float32");
    if (is_contiguous()) {
        float* ptr = data_ptr<float>();
        for (int64_t i = 0; i < numel_; ++i) {
            ptr[i] = val;
        }
    } else {
        // Multi-dimensional non-contiguous assignment
        std::vector<int64_t> coords(ndim(), 0);
        float* base = static_cast<float*>(storage_->data);
        for (int64_t i = 0; i < numel_; ++i) {
            int64_t elem_offset = offset_;
            for (int64_t d = 0; d < ndim(); ++d) {
                elem_offset += coords[d] * strides_[d];
            }
            base[elem_offset] = val;

            for (int64_t d = ndim() - 1; d >= 0; --d) {
                coords[d]++;
                if (coords[d] < shape_[d]) break;
                coords[d] = 0;
            }
        }
    }
    return *this;
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    int64_t infer_idx = -1;
    int64_t prod = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            VAI_CHECK(infer_idx == -1, "Only one dimension can be inferred (-1)");
            infer_idx = static_cast<int64_t>(i);
        } else {
            VAI_CHECK(new_shape[i] > 0, "Invalid dimension size");
            prod *= new_shape[i];
        }
    }

    std::vector<int64_t> resolved_shape = new_shape;
    if (infer_idx != -1) {
        VAI_CHECK(numel_ % prod == 0, "Inferred dimension does not divide total elements");
        resolved_shape[infer_idx] = numel_ / prod;
    } else {
        VAI_CHECK(prod == numel_, "Reshape size mismatch");
    }

    if (is_contiguous()) {
        // Zero-copy view
        return Tensor(storage_, offset_, resolved_shape, compute_contiguous_strides(resolved_shape), dtype_, device_);
    } else {
        // Copy to contiguous then reshape
        return contiguous().reshape(resolved_shape);
    }
}

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    VAI_CHECK(is_contiguous(), "Tensor must be contiguous to call view()");
    return reshape(new_shape);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    if (dim0 < 0) dim0 += ndim();
    if (dim1 < 0) dim1 += ndim();
    VAI_CHECK(dim0 >= 0 && dim0 < ndim(), "dim0 out of range");
    VAI_CHECK(dim1 >= 0 && dim1 < ndim(), "dim1 out of range");

    std::vector<int64_t> new_shape = shape_;
    std::vector<int64_t> new_strides = strides_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    // Zero-copy view!
    return Tensor(storage_, offset_, new_shape, new_strides, dtype_, device_);
}

Tensor Tensor::permute(const std::vector<int64_t>& dims) const {
    VAI_CHECK(static_cast<int64_t>(dims.size()) == ndim(), "Permute dimensions size mismatch");
    std::vector<int64_t> new_shape(ndim());
    std::vector<int64_t> new_strides(ndim());
    std::vector<bool> seen(ndim(), false);

    for (int64_t i = 0; i < ndim(); ++i) {
        int64_t d = dims[i];
        if (d < 0) d += ndim();
        VAI_CHECK(d >= 0 && d < ndim() && !seen[d], "Invalid permutation dims");
        seen[d] = true;
        new_shape[i] = shape_[d];
        new_strides[i] = strides_[d];
    }

    // Zero-copy view!
    return Tensor(storage_, offset_, new_shape, new_strides, dtype_, device_);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    if (dim < 0) dim += ndim();
    VAI_CHECK(dim >= 0 && dim < ndim(), "dim out of range");
    VAI_CHECK(step > 0, "Step must be positive");

    int64_t dim_size = shape_[dim];
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;
    start = std::max<int64_t>(0, std::min<int64_t>(start, dim_size));
    end = std::max<int64_t>(start, std::min<int64_t>(end, dim_size));

    int64_t new_dim_size = (end - start + step - 1) / step;
    std::vector<int64_t> new_shape = shape_;
    std::vector<int64_t> new_strides = strides_;

    new_shape[dim] = new_dim_size;
    new_strides[dim] = strides_[dim] * step;
    int64_t new_offset = offset_ + start * strides_[dim];

    // Zero-copy view!
    return Tensor(storage_, new_offset, new_shape, new_strides, dtype_, device_);
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;

    Tensor result(shape_, dtype_, device_);
    VAI_CHECK(dtype_ == DType::Float32, "contiguous copy currently Float32");

    float* dst = result.data_ptr<float>();
    const float* src_base = static_cast<const float*>(storage_->data);

    std::vector<int64_t> coords(ndim(), 0);
    for (int64_t i = 0; i < numel_; ++i) {
        int64_t elem_offset = offset_;
        for (int64_t d = 0; d < ndim(); ++d) {
            elem_offset += coords[d] * strides_[d];
        }
        dst[i] = src_base[elem_offset];

        for (int64_t d = ndim() - 1; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < shape_[d]) break;
            coords[d] = 0;
        }
    }
    return result;
}

Tensor Tensor::clone() const {
    Tensor result(shape_, dtype_, device_);
    if (is_contiguous()) {
        std::memcpy(result.data_ptr(), data_ptr(), nbytes());
    } else {
        result = contiguous();
    }
    return result;
}

Tensor Tensor::to(Device new_device) const {
    if (device_ == new_device) return *this;
    
    Tensor a = contiguous();
    Tensor out(shape_, dtype_, new_device);
    
    // For now, since Metal and CPU share memory in our prototype (we used malloc/MTLResourceStorageModeShared), 
    // a simple memcpy works. Real discrete GPUs would need cudaMemcpy / MTLBuffer replaceBytes.
    std::memcpy(out.data_ptr(), a.data_ptr(), a.nbytes());
    
    return out;
}

Tensor Tensor::add(const Tensor& other) const {
    VAI_CHECK(shape_ == other.shape_, "Shape mismatch for add");
    VAI_CHECK(dtype_ == other.dtype_ && dtype_ == DType::Float32, "add currently supports Float32");
    
    Tensor a = contiguous();
    Tensor b = other.contiguous();
    Tensor out(shape_, dtype_, device_);

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int64_t n = numel_;

    // Call SIMD C kernel
    kernel_add_f32(a_ptr, b_ptr, out_ptr, n);

    return out;
}

Tensor Tensor::sub(const Tensor& other) const {
    VAI_CHECK(shape_ == other.shape_, "Shape mismatch for sub");
    VAI_CHECK(dtype_ == other.dtype_ && dtype_ == DType::Float32, "sub currently supports Float32");

    Tensor a = contiguous();
    Tensor b = other.contiguous();
    Tensor out(shape_, dtype_, device_);

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int64_t n = numel_;

    // Call SIMD C kernel
    kernel_sub_f32(a_ptr, b_ptr, out_ptr, n);

    return out;
}

Tensor Tensor::mul(const Tensor& other) const {
    VAI_CHECK(shape_ == other.shape_, "Shape mismatch for mul");
    VAI_CHECK(dtype_ == other.dtype_ && dtype_ == DType::Float32, "mul currently supports Float32");

    Tensor a = contiguous();
    Tensor b = other.contiguous();
    Tensor out(shape_, dtype_, device_);

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int64_t n = numel_;

    // Call SIMD C kernel
    kernel_mul_f32(a_ptr, b_ptr, out_ptr, n);

    return out;
}

Tensor Tensor::div(const Tensor& other) const {
    VAI_CHECK(shape_ == other.shape_, "Shape mismatch for div");
    VAI_CHECK(dtype_ == other.dtype_ && dtype_ == DType::Float32, "div currently supports Float32");

    Tensor a = contiguous();
    Tensor b = other.contiguous();
    Tensor out(shape_, dtype_, device_);

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int64_t n = numel_;

    // Call SIMD C kernel
    kernel_div_f32(a_ptr, b_ptr, out_ptr, n);

    return out;
}

Tensor Tensor::add_scalar(float val) const {
    Tensor a = contiguous();
    Tensor out(shape_, dtype_, device_);
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int64_t n = numel_;

    // Call SIMD C kernel
    kernel_add_scalar_f32(a_ptr, val, out_ptr, n);

    return out;
}

Tensor Tensor::mul_scalar(float val) const {
    Tensor a = contiguous();
    Tensor out(shape_, dtype_, device_);
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int64_t n = numel_;

    // Call SIMD C kernel
    kernel_mul_scalar_f32(a_ptr, val, out_ptr, n);

    return out;
}

Tensor Tensor::matmul(const Tensor& other) const {
    VAI_CHECK(ndim() == 2 && other.ndim() == 2, "MatMul currently requires 2D tensors");
    VAI_CHECK(shape_[1] == other.shape_[0], "MatMul inner dimension mismatch");
    VAI_CHECK(dtype_ == DType::Float32 && other.dtype_ == DType::Float32, "MatMul requires Float32");

    int64_t M = shape_[0];
    int64_t K = shape_[1];
    int64_t N = other.shape_[1];

    Tensor a = contiguous();
    Tensor b = other.contiguous();
    Tensor out({M, N}, dtype_, device_);
    out.zero_();

    // Route based on device
    if (device_.is_metal()) {
#ifdef __APPLE__
        velocityai::metal::matmul_metal(a, b, out);
#else
        throw std::runtime_error("Metal backend not compiled on this platform.");
#endif
    } else if (device_.is_cuda()) {
        throw std::runtime_error("CUDA backend not implemented — this build targets Metal/CPU only");
    } else {
        // CPU execution
        const float* A = a.data_ptr<float>();
        const float* B = b.data_ptr<float>();
        float* C = out.data_ptr<float>();

        // Call SIMD C kernel
        kernel_matmul_f32(A, B, C, M, K, N);
    }

    return out;
}

Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    VAI_CHECK(dtype_ == DType::Float32, "sum currently supports Float32");
    Tensor c = contiguous();
    const float* ptr = c.data_ptr<float>();

    if (dim == -1 && !keepdim) {
        // Full reduction
        Tensor res({1}, dtype_, device_);
        // Call SIMD C kernel
        kernel_sum_f32(ptr, res.data_ptr<float>(), numel_);
        return res;
    }

    if (dim < 0) dim += ndim();
    VAI_CHECK(dim >= 0 && dim < ndim(), "dim out of range");

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim(); ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape_[i]);
        }
    }
    if (out_shape.empty()) out_shape = {1};

    Tensor out = Tensor::zeros(out_shape, dtype_, device_);
    float* out_ptr = out.data_ptr<float>();

    int64_t dim_size = shape_[dim];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape_[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim(); ++i) inner_size *= shape_[i];

    #pragma omp parallel for collapse(2) if(outer_size * inner_size > 1024)
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t in = 0; in < inner_size; ++in) {
            double s = 0.0;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t src_idx = o * (dim_size * inner_size) + d * inner_size + in;
                s += ptr[src_idx];
            }
            int64_t dst_idx = o * inner_size + in;
            out_ptr[dst_idx] = static_cast<float>(s);
        }
    }
    return out;
}

Tensor Tensor::mean(int64_t dim, bool keepdim) const {
    Tensor s = sum(dim, keepdim);
    int64_t count = (dim == -1) ? numel_ : shape(dim);
    return s.mul_scalar(1.0f / static_cast<float>(count));
}

Tensor Tensor::relu() const {
    VAI_CHECK(dtype_ == DType::Float32, "relu currently supports Float32");
    Tensor c = contiguous();
    Tensor out(shape_, dtype_, device_);
    const float* in_ptr = c.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int64_t n = numel_;

    // Call SIMD C kernel
    kernel_relu_f32(in_ptr, out_ptr, n);

    return out;
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "velocityai.tensor(";
    if (numel_ == 0) {
        oss << "[]";
    } else if (ndim() <= 2) {
        Tensor c = contiguous();
        const float* ptr = c.data_ptr<float>();
        if (ndim() == 1) {
            oss << "[";
            for (int64_t i = 0; i < std::min<int64_t>(numel_, 10); ++i) {
                if (i > 0) oss << ", ";
                oss << std::fixed << std::setprecision(4) << ptr[i];
            }
            if (numel_ > 10) oss << ", ...";
            oss << "]";
        } else if (ndim() == 2) {
            oss << "[\n";
            int64_t rows = std::min<int64_t>(shape_[0], 8);
            int64_t cols = std::min<int64_t>(shape_[1], 8);
            for (int64_t r = 0; r < rows; ++r) {
                oss << "  [";
                for (int64_t c_idx = 0; c_idx < cols; ++c_idx) {
                    if (c_idx > 0) oss << ", ";
                    oss << std::fixed << std::setprecision(4) << ptr[r * shape_[1] + c_idx];
                }
                if (shape_[1] > cols) oss << ", ...";
                oss << "]";
                if (r < rows - 1) oss << ",\n";
            }
            if (shape_[0] > rows) oss << ",\n  ...";
            oss << "\n]";
        }
    } else {
        oss << "[shape=";
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i > 0) oss << "x";
            oss << shape_[i];
        }
        oss << "]";
    }
    oss << ", shape=(";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    if (shape_.size() == 1) oss << ",";
    oss << "), dtype=" << dtype_name(dtype_) << ", device='" << device_.str() << "')";
    return oss.str();
}

} // namespace velocityai
