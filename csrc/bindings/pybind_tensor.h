#pragma once

#include <pybind11/pybind11.h>

namespace velocityai {
    void init_tensor_bindings(pybind11::module_& m);
    void init_memory_bindings(pybind11::module_& m);
}
