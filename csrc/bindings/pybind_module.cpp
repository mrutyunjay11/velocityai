#include <pybind11/pybind11.h>
#include "csrc/bindings/pybind_tensor.h"

PYBIND11_MODULE(_velocityai_c, m) {
    m.doc() = "VelocityAI High-Performance C++ Core Engine";

    velocityai::init_tensor_bindings(m);
    velocityai::init_memory_bindings(m);
}
