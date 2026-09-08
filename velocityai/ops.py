from typing import Union
import numpy as np
from velocityai.tensor import Tensor
from velocityai.compiler.tracer import get_active_tracer
import velocityai.autograd as ag

def matmul(input: Tensor, other: Tensor) -> Tensor:
    """Matrix multiplication of two 2D tensors."""
    if not isinstance(input, Tensor) or not isinstance(other, Tensor):
        raise TypeError("matmul operands must be velocityai.Tensor")
        
    out = ag.MatMul.apply(input, other)
    tracer = get_active_tracer()
    if tracer:
        tracer.record_op("MatMul", [input, other], out)
    return out

def relu(input: Tensor) -> Tensor:
    """Rectified Linear Unit activation."""
    if not isinstance(input, Tensor):
        raise TypeError("relu operand must be velocityai.Tensor")
    out = ag.ReLU.apply(input)
    tracer = get_active_tracer()
    if tracer:
        tracer.record_op("ReLU", [input], out)
    return out

def add(input: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """Element-wise addition."""
    out = ag.Add.apply(input, other)
    tracer = get_active_tracer()
    if tracer:
        tracer.record_op("Add", [input, other], out)
    return out

def sub(input: Tensor, other: Tensor) -> Tensor:
    """Element-wise subtraction."""
    out = ag.Sub.apply(input, other)
    tracer = get_active_tracer()
    if tracer:
        tracer.record_op("Sub", [input, other], out)
    return out

def mul(input: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """Element-wise multiplication."""
    out = ag.Mul.apply(input, other)
    tracer = get_active_tracer()
    if tracer:
        tracer.record_op("Mul", [input, other], out)
    return out

def div(input: Tensor, other: Union[Tensor, float, int]) -> Tensor:
    """Element-wise division."""
    if isinstance(other, (int, float)):
        return Tensor._wrap(input._c.mul_scalar(1.0 / float(other)))
    return Tensor._wrap(input._c.div(other._c))

def sum(input: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Sum reduction along specified dimension or full tensor."""
    out = ag.Sum.apply(input)
    tracer = get_active_tracer()
    if tracer:
        tracer.record_op("Sum", [input], out)
    return out

def mean(input: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Mean reduction along specified dimension or full tensor."""
    return input.mean(dim=dim, keepdim=keepdim)

def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return ag.CrossEntropy.apply(logits, targets)

def allclose(a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Check if two tensors have all close elements within tolerance."""
    return np.allclose(a.numpy(), b.numpy(), rtol=rtol, atol=atol)

__all__ = [
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
]
