"""VelocityAI: High-Level to Low-Level AI Acceleration Engine."""

__version__ = "0.1.0"

from velocityai.dtype import (
    DType,
    float32,
    float16,
    bfloat16,
    int32,
    int16,
    int8,
    int4,
    bool,
    f32,
    f16,
    bf16,
    i32,
    i8,
)

from velocityai.device import (
    Device,
    device,
    is_metal_available,
    is_cuda_available,
)

from velocityai.tensor import (
    Tensor,
    tensor,
    empty,
    zeros,
    ones,
    full,
    randn,
    from_numpy,
)

from velocityai.ops import (
    matmul,
    relu,
    add,
    sub,
    mul,
    div,
    sum,
    mean,
    allclose,
)

from velocityai.compiler.jit import compile
from _velocityai_c import memory
import velocityai.nn as nn
import velocityai.optim as optim
import velocityai.data as data
import velocityai.distributed as distributed
from velocityai.weight_loader import load_huggingface_model
from velocityai.tokenizer import Tokenizer
from velocityai.generate import generate

__all__ = [
    "__version__",
    "Tensor",
    "tensor",
    "empty",
    "zeros",
    "ones",
    "full",
    "randn",
    "from_numpy",
    "DType",
    "float32",
    "float16",
    "bfloat16",
    "int32",
    "int16",
    "int8",
    "int4",
    "bool",
    "f32",
    "f16",
    "bf16",
    "i32",
    "i8",
    "Device",
    "device",
    "is_metal_available",
    "is_cuda_available",
    "matmul",
    "relu",
    "add",
    "sub",
    "mul",
    "div",
    "sum",
    "mean",
    "cross_entropy_loss",
    "allclose",
    "memory",
    "compile",
    "nn",
    "optim",
    "data",
    "distributed",
]
