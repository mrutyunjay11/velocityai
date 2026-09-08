#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace velocityai {

enum class DType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int16 = 4,
    Int8 = 5,
    Int4 = 6,
    Bool = 7
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Int32: return 4;
        case DType::Int16: return 2;
        case DType::Int8: return 1;
        case DType::Int4: return 1; // packed representation
        case DType::Bool: return 1;
        default:
            throw std::invalid_argument("Unknown DType");
    }
}

inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int32: return "int32";
        case DType::Int16: return "int16";
        case DType::Int8: return "int8";
        case DType::Int4: return "int4";
        case DType::Bool: return "bool";
        default: return "unknown";
    }
}

inline DType dtype_from_string(const std::string& name) {
    if (name == "float32" || name == "f32" || name == "float") return DType::Float32;
    if (name == "float16" || name == "f16" || name == "half") return DType::Float16;
    if (name == "bfloat16" || name == "bf16") return DType::BFloat16;
    if (name == "int32" || name == "i32" || name == "int") return DType::Int32;
    if (name == "int16" || name == "i16" || name == "short") return DType::Int16;
    if (name == "int8" || name == "i8") return DType::Int8;
    if (name == "int4" || name == "i4") return DType::Int4;
    if (name == "bool" || name == "boolean") return DType::Bool;
    throw std::invalid_argument("Invalid dtype name: " + name);
}

} // namespace velocityai
