cmake_minimum_required(VERSION 3.18)
project(velocityai LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /W3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -Wall -Wno-unknown-pragmas")
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.15")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

# Find Python and Pybind11
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Core library sources
set(CORE_SOURCES
    csrc/core/memory_pool.cpp
    csrc/core/tensor.cpp
    csrc/kernels/cpu/math_kernels.cpp
    csrc/kernels/cpu/reduce.cpp
    csrc/kernels/cpu/softmax.cpp
    csrc/kernels/cpu/normalization.cpp
    csrc/kernels/cpu/attention.cpp
    csrc/kernels/cpu/conv2d.cpp
    csrc/kernels/cpu/loss.cpp
    csrc/kernels/cpu/embedding.cpp
    csrc/bindings/pybind_tensor.cpp
    csrc/bindings/pybind_module.cpp
)

# Build Python C-extension
pybind11_add_module(_velocityai_c MODULE ${CORE_SOURCES})
target_include_directories(_velocityai_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Install configuration
install(TARGETS _velocityai_c DESTINATION velocityai)
