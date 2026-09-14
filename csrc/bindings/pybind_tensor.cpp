#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "csrc/core/tensor.h"
#include "csrc/core/memory_pool.h"
#include "csrc/kernels/cpu/cpu_kernels.h"
#include "csrc/core/llama_decoder.h"

namespace py = pybind11;

namespace velocityai {

void init_tensor_bindings(py::module_& m) {
    // Expose DType enum
    py::enum_<DType>(m, "DType")
        .value("float32", DType::Float32)
        .value("float16", DType::Float16)
        .value("bfloat16", DType::BFloat16)
        .value("int32", DType::Int32)
        .value("int16", DType::Int16)
        .value("int8", DType::Int8)
        .value("int4", DType::Int4)
        .value("bool", DType::Bool)
        .export_values();

    // Expose Device class
    py::class_<Device>(m, "Device")
        .def(py::init<DeviceType, int>(), py::arg("type"), py::arg("index") = 0)
        .def_static("from_string", &Device::from_string)
        .def_property_readonly("type", [](const Device& d) { return static_cast<int>(d.type); })
        .def_property_readonly("index", [](const Device& d) { return d.index; })
        .def("is_cpu", &Device::is_cpu)
        .def("is_cuda", &Device::is_cuda)
        .def("is_metal", &Device::is_metal)
        .def("__repr__", &Device::str)
        .def("__str__", &Device::str);

    // Expose Tensor with buffer protocol
    py::class_<Tensor> tensor(m, "_Tensor", py::buffer_protocol());

    tensor.def_buffer([](Tensor& t) -> py::buffer_info {
        VAI_CHECK(t.device().is_cpu(), "Cannot convert non-CPU tensor to NumPy buffer");
        std::string format;
        if (t.dtype() == DType::Float32) format = py::format_descriptor<float>::format();
        else if (t.dtype() == DType::Float16) format = "e";
        else if (t.dtype() == DType::Int32) format = py::format_descriptor<int32_t>::format();
        else if (t.dtype() == DType::Int8) format = py::format_descriptor<int8_t>::format();
        else if (t.dtype() == DType::Bool) format = py::format_descriptor<bool>::format();
        else format = "B"; // fallback raw bytes

        std::vector<ssize_t> py_shape(t.shape().begin(), t.shape().end());
        std::vector<ssize_t> py_strides;
        py_strides.reserve(t.strides().size());
        for (auto s : t.strides()) {
            py_strides.push_back(s * static_cast<ssize_t>(t.itemsize()));
        }

        return py::buffer_info(
            t.data_ptr(),
            static_cast<ssize_t>(t.itemsize()),
            format,
            static_cast<ssize_t>(t.ndim()),
            py_shape,
            py_strides
        );
    });

    tensor.def(py::init<const std::vector<int64_t>&, DType, Device>(),
               py::arg("shape"), py::arg("dtype") = DType::Float32, py::arg("device") = Device::CPU())
        .def_static("empty", &Tensor::empty, py::arg("shape"), py::arg("dtype") = DType::Float32, py::arg("device") = Device::CPU())
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("dtype") = DType::Float32, py::arg("device") = Device::CPU())
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("dtype") = DType::Float32, py::arg("device") = Device::CPU())
        .def_static("full", &Tensor::full, py::arg("shape"), py::arg("value"), py::arg("dtype") = DType::Float32, py::arg("device") = Device::CPU())
        .def_static("randn", &Tensor::randn, py::arg("shape"), py::arg("mean") = 0.0f, py::arg("std") = 1.0f, py::arg("dtype") = DType::Float32, py::arg("device") = Device::CPU())
        .def_static("from_numpy", [](py::array array, bool copy) {
            py::buffer_info info = array.request();
            std::string dt_str = py::str(array.attr("dtype")).cast<std::string>();
            DType dt = DType::Float32;
            if (dt_str == "float32") dt = DType::Float32;
            else if (dt_str == "float16") dt = DType::Float16;
            else if (dt_str == "int32") dt = DType::Int32;
            else if (dt_str == "int8") dt = DType::Int8;
            else if (dt_str == "bool") dt = DType::Bool;
            else {
                throw TypeError("Unsupported NumPy dtype: " + dt_str + ". Please convert to float32 or float16.");
            }

            std::vector<int64_t> shape;
            for (auto s : info.shape) shape.push_back(static_cast<int64_t>(s));

            if (copy) {
                return Tensor::from_blob(info.ptr, shape, dt, Device::CPU(), true);
            } else {
                // Keep reference to numpy array to prevent deallocation
                py::object base = array;
                size_t nbytes = Tensor::compute_numel(shape) * dtype_size(dt);
                auto storage = std::make_shared<Storage>(info.ptr, nbytes, Device::CPU(), false);
                std::vector<int64_t> strides;
                for (auto s : info.strides) {
                    strides.push_back(static_cast<int64_t>(s) / dtype_size(dt));
                }
                return Tensor(storage, 0, shape, strides, dt, Device::CPU());
            }
        }, py::arg("array"), py::arg("copy") = false)
        .def_property_readonly("shape", [](const Tensor& t) -> std::vector<int64_t> { return t.shape(); })
        .def_property_readonly("strides", [](const Tensor& t) -> std::vector<int64_t> { return t.strides(); })
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("device", &Tensor::device)
        .def_property_readonly("itemsize", &Tensor::itemsize)
        .def_property_readonly("nbytes", &Tensor::nbytes)
        .def_property_readonly("is_contiguous", &Tensor::is_contiguous)
        .def_property("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def("reshape", &Tensor::reshape, py::arg("new_shape"))
        .def("view", &Tensor::view, py::arg("new_shape"))
        .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
        .def("permute", &Tensor::permute, py::arg("dims"))
        .def("slice", &Tensor::slice, py::arg("dim"), py::arg("start"), py::arg("end"), py::arg("step") = 1)
        .def("contiguous", &Tensor::contiguous)
        .def("clone", &Tensor::clone)
        .def("to", [](const Tensor& t, const std::string& dev_str) {
            return t.to(Device::from_string(dev_str));
        }, py::arg("device"))
        .def("zero_", &Tensor::zero_)
        .def("fill_", &Tensor::fill_, py::arg("value"))
        .def("add", &Tensor::add, py::arg("other"))
        .def("sub", &Tensor::sub, py::arg("other"))
        .def("mul", &Tensor::mul, py::arg("other"))
        .def("div", &Tensor::div, py::arg("other"))
        .def("matmul", &Tensor::matmul, py::arg("other"))
        .def("add_scalar", &Tensor::add_scalar, py::arg("value"))
        .def("mul_scalar", &Tensor::mul_scalar, py::arg("value"))
        .def("sum", &Tensor::sum, py::arg("dim") = -1, py::arg("keepdim") = false)
        .def("mean", &Tensor::mean, py::arg("dim") = -1, py::arg("keepdim") = false)
        .def("relu", &Tensor::relu)
        .def("__repr__", &Tensor::to_string)
        .def("__str__", &Tensor::to_string)
        .def("__add__", &Tensor::add)
        .def("__add__", &Tensor::add_scalar)
        .def("__sub__", &Tensor::sub)
        .def("__mul__", &Tensor::mul)
        .def("__mul__", &Tensor::mul_scalar)
        .def("__truediv__", &Tensor::div)
        .def("__matmul__", &Tensor::matmul)
        .def("__len__", [](const Tensor& t) {
            VAI_CHECK(t.ndim() > 0, "len() of a 0-d tensor is undefined");
            return t.shape(0);
        });

    m.def("rms_norm", [](const Tensor& x, const Tensor& weight, float eps) {
        Tensor x_c = x.contiguous();
        Tensor w_c = weight.contiguous();
        int64_t cols = w_c.numel();
        int64_t rows = x_c.numel() / cols;
        Tensor out = Tensor::empty(x_c.shape(), x_c.dtype(), x_c.device());
        kernel_rmsnorm_f32(x_c.data_ptr<float>(), w_c.data_ptr<float>(), out.data_ptr<float>(), rows, cols, eps);
        return out;
    }, py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-5f);

    m.def("swiglu", [](const Tensor& gate, const Tensor& up) {
        Tensor g_c = gate.contiguous();
        Tensor u_c = up.contiguous();
        VAI_CHECK(g_c.shape() == u_c.shape(), "swiglu shapes must match");
        Tensor out = Tensor::empty(g_c.shape(), g_c.dtype(), g_c.device());
        kernel_swiglu_f32(g_c.data_ptr<float>(), u_c.data_ptr<float>(), out.data_ptr<float>(), g_c.numel());
        return out;
    }, py::arg("gate"), py::arg("up"));

    m.def("fused_attention_decode", [](
        Tensor& q,
        Tensor& k,
        const Tensor& v,
        Tensor& k_cache,
        Tensor& v_cache,
        const Tensor& cos,
        const Tensor& sin,
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t head_dim,
        int64_t max_seq_len,
        int64_t start_pos
    ) {
        Tensor out = Tensor::empty({1, 1, num_heads * head_dim}, DType::Float32, Device::CPU());
        kernel_attention_decode_f32(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            k_cache.data_ptr<float>(),
            v_cache.data_ptr<float>(),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            out.data_ptr<float>(),
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            start_pos
        );
        return out;
    }, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("k_cache"), py::arg("v_cache"),
       py::arg("cos"), py::arg("sin"), py::arg("num_heads"), py::arg("num_kv_heads"),
       py::arg("head_dim"), py::arg("max_seq_len"), py::arg("start_pos"));

    m.def("fused_attention_decode_qkv", [](
        Tensor& qkv,
        Tensor& k_cache,
        Tensor& v_cache,
        const Tensor& cos,
        const Tensor& sin,
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t head_dim,
        int64_t max_seq_len,
        int64_t start_pos
    ) {
        Tensor out = Tensor::empty({1, 1, num_heads * head_dim}, DType::Float32, Device::CPU());
        float* qkv_ptr = qkv.data_ptr<float>();
        int64_t q_dim = num_heads * head_dim;
        int64_t k_dim = num_kv_heads * head_dim;
        kernel_attention_decode_f32(
            qkv_ptr,
            qkv_ptr + q_dim,
            qkv_ptr + q_dim + k_dim,
            k_cache.data_ptr<float>(),
            v_cache.data_ptr<float>(),
            cos.data_ptr<float>(),
            sin.data_ptr<float>(),
            out.data_ptr<float>(),
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            start_pos
        );
        return out;
    }, py::arg("qkv"), py::arg("k_cache"), py::arg("v_cache"),
       py::arg("cos"), py::arg("sin"), py::arg("num_heads"), py::arg("num_kv_heads"),
       py::arg("head_dim"), py::arg("max_seq_len"), py::arg("start_pos"));

    m.def("gemv", [](const Tensor& x, const Tensor& weight) -> Tensor {
        Tensor x_c = x.contiguous();
        Tensor w_c = weight.contiguous();
        int64_t K = x_c.shape().back();
        int64_t N = w_c.shape()[0];
        VAI_CHECK(w_c.shape()[1] == K, "GEMV inner dimension mismatch");

        std::vector<int64_t> out_shape = x_c.shape();
        out_shape.back() = N;
        Tensor out = Tensor::empty(out_shape, DType::Float32, Device::CPU());

        const float* x_ptr = x_c.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        if (w_c.dtype() == DType::Float16) {
            kernel_gemv_fp16(x_ptr, w_c.data_ptr(), out_ptr, K, N);
        } else if (w_c.dtype() == DType::Float32) {
            kernel_gemv_f32(x_ptr, w_c.data_ptr<float>(), out_ptr, K, N);
        } else {
            throw TypeError("Unsupported weight dtype for GEMV (must be float16 or float32)");
        }
        return out;
    }, py::arg("x"), py::arg("weight"));

    m.def("gemv_swiglu", [](const Tensor& x, const Tensor& weight_gate_up) -> Tensor {
        Tensor x_c = x.contiguous();
        Tensor w_c = weight_gate_up.contiguous();
        int64_t K = x_c.shape().back();
        int64_t total_intermediate = w_c.shape()[0];
        VAI_CHECK(total_intermediate % 2 == 0, "Gate-Up weight rows must be 2 * intermediate");
        int64_t intermediate = total_intermediate / 2;
        VAI_CHECK(w_c.shape()[1] == K, "GEMV inner dimension mismatch");

        std::vector<int64_t> out_shape = x_c.shape();
        out_shape.back() = intermediate;
        Tensor out = Tensor::empty(out_shape, DType::Float32, Device::CPU());

        const float* x_ptr = x_c.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        if (w_c.dtype() == DType::Float16) {
            kernel_gemv_fp16_fused_swiglu(x_ptr, w_c.data_ptr(), out_ptr, K, intermediate);
        } else if (w_c.dtype() == DType::Float32) {
            kernel_gemv_f32_fused_swiglu(x_ptr, w_c.data_ptr<float>(), out_ptr, K, intermediate);
        } else {
            throw TypeError("Unsupported weight dtype for GEMV SwiGLU (must be float16 or float32)");
        }
        return out;
    }, py::arg("x"), py::arg("weight_gate_up"));

    m.def("linear", [](const Tensor& x, const Tensor& weight) -> Tensor {
        Tensor x_c = x.contiguous();
        Tensor w_c = weight.contiguous();
        int64_t K = x_c.shape().back();
        int64_t N = w_c.shape()[0];
        VAI_CHECK(w_c.shape()[1] == K, "Linear dimension mismatch: weight cols must equal x last dim");

        int64_t M = x_c.numel() / K;
        std::vector<int64_t> out_shape = x_c.shape();
        out_shape.back() = N;
        Tensor out = Tensor::empty(out_shape, DType::Float32, Device::CPU());

        const float* x_ptr = x_c.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        if (M == 1) {
            if (w_c.dtype() == DType::Float16) {
                kernel_gemv_fp16(x_ptr, w_c.data_ptr(), out_ptr, K, N);
            } else if (w_c.dtype() == DType::Float32) {
                kernel_gemv_f32(x_ptr, w_c.data_ptr<float>(), out_ptr, K, N);
            } else {
                throw TypeError("Unsupported weight dtype for linear");
            }
        } else {
            if (w_c.dtype() == DType::Float32) {
                kernel_matmul_transposed_f32(x_ptr, w_c.data_ptr<float>(), out_ptr, M, K, N);
            } else if (w_c.dtype() == DType::Float16) {
                kernel_gemm_fp16(x_ptr, w_c.data_ptr(), out_ptr, M, K, N);
            } else {
                throw TypeError("Unsupported weight dtype for multi-token linear");
            }
        }
        return out;
    }, py::arg("x"), py::arg("weight"));

    py::class_<FastLlamaDecoder>(m, "FastLlamaDecoder")
        .def(py::init<
            int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, float,
            const Tensor&, const Tensor&, const Tensor&
        >(),
        py::arg("vocab_size"),
        py::arg("dim"),
        py::arg("num_layers"),
        py::arg("num_heads"),
        py::arg("num_kv_heads"),
        py::arg("head_dim"),
        py::arg("intermediate_size"),
        py::arg("max_seq_len"),
        py::arg("eps"),
        py::arg("rope_theta"),
        py::arg("embed_tokens_weight"),
        py::arg("norm_weight"),
        py::arg("lm_head_weight"))
        .def("add_layer", &FastLlamaDecoder::add_layer,
            py::arg("input_layernorm_weight"),
            py::arg("qkv_weight"),
            py::arg("o_weight"),
            py::arg("post_attn_layernorm_weight"),
            py::arg("gate_up_weight"),
            py::arg("down_weight"),
            py::arg("k_cache"),
            py::arg("v_cache"),
            py::arg("qkv_bias") = nullptr)
        .def("decode_step", &FastLlamaDecoder::decode_step, py::arg("token_id"), py::arg("start_pos"))
        .def("decode_step_logits", &FastLlamaDecoder::decode_step_logits, py::arg("token_id"), py::arg("start_pos"))
        .def("prefill", &FastLlamaDecoder::prefill, py::arg("prompt_tokens"))
        .def("prefill_logits", &FastLlamaDecoder::prefill_logits, py::arg("prompt_tokens"))
        .def("set_sparse_mode", &FastLlamaDecoder::set_sparse_mode, py::arg("enabled"), py::arg("threshold") = -3.5f)
        .def_property_readonly("sparse_mode", &FastLlamaDecoder::get_sparse_mode)
        .def_property_readonly("sparse_threshold", &FastLlamaDecoder::get_sparse_threshold)
        .def("is_cache_populated", &FastLlamaDecoder::is_cache_populated);
}

void init_memory_bindings(py::module_& m) {
    py::module_ mem = m.def_submodule("memory", "VelocityAI memory management");
    mem.def("empty_cache", []() {
        MemoryPool::instance().empty_cache();
    }, "Release all cached memory in the pool back to OS");
    mem.def("allocated_bytes", []() {
        return MemoryPool::instance().allocated_bytes();
    }, "Get total currently allocated bytes");
    mem.def("cached_bytes", []() {
        return MemoryPool::instance().cached_bytes();
    }, "Get total cached bytes in memory pool");
    mem.def("peak_bytes", []() {
        return MemoryPool::instance().peak_bytes();
    }, "Get peak memory bytes allocated");
}

} // namespace velocityai
