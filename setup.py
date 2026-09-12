from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os
import setuptools
from setuptools.command.build_ext import build_ext as _build_ext
import distutils.unixccompiler

# Add .mm support
distutils.unixccompiler.UnixCCompiler.src_extensions.append('.mm')
distutils.unixccompiler.UnixCCompiler.language_map['.mm'] = 'objc'

__version__ = "0.1.0"

extra_compile_args = [
    "-std=c++17",
    "-O3",
    "-ffast-math",
    "-Wno-unknown-pragmas",
]

extra_link_args = []

if sys.platform == "darwin":
    extra_compile_args += ["-mmacosx-version-min=10.15"]
    extra_link_args += ["-framework", "Metal", "-framework", "Foundation", "-framework", "Accelerate"]
elif sys.platform.startswith("linux"):
    extra_compile_args += ["-march=native", "-fopenmp"]
elif sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17", "/openmp"]

ext_modules = [
    Pybind11Extension(
        "_velocityai_c",
        sources=[
            "csrc/core/memory_pool.cpp",
            "csrc/core/tensor.cpp",
            "csrc/core/llama_decoder.cpp",
            "csrc/kernels/cpu/math_kernels.cpp",
            "csrc/kernels/cpu/reduce.cpp",
            "csrc/kernels/cpu/softmax.cpp",
            "csrc/kernels/cpu/normalization.cpp",
            "csrc/kernels/cpu/attention.cpp",
            "csrc/kernels/cpu/conv2d.cpp",
            "csrc/kernels/cpu/loss.cpp",
            "csrc/kernels/cpu/embedding.cpp",

            "csrc/kernels/metal/metal_backend.mm",
            "csrc/bindings/pybind_tensor.cpp",
            "csrc/bindings/pybind_module.cpp",
        ],
        include_dirs=[
            os.path.abspath("."),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
]

setup(
    name="velocityai",
    version=__version__,
    author="VelocityAI Team",
    description="High-Level to Low-Level AI Acceleration Engine",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
