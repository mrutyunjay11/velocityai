from typing import Sequence, Union, Optional
import numpy as np
from _velocityai_c import _Tensor, Device
from velocityai.dtype import float32, DType

class Tensor:
    """VelocityAI High-Performance Multi-Dimensional Tensor."""

    def __init__(self, c_tensor: _Tensor, requires_grad: bool = False):
        self._c = c_tensor
        self._c.requires_grad = requires_grad
        self.grad: Optional['Tensor'] = None
        self._grad_fn = None

    @classmethod
    def _wrap(cls, c_t: _Tensor, requires_grad: bool = False) -> 'Tensor':
        return cls(c_t, requires_grad=requires_grad)

    @property
    def shape(self) -> Sequence[int]:
        return tuple(self._c.shape)

    @property
    def strides(self) -> Sequence[int]:
        return tuple(self._c.strides)

    @property
    def ndim(self) -> int:
        return self._c.ndim

    @property
    def numel(self) -> int:
        return self._c.numel

    @property
    def dtype(self) -> DType:
        return self._c.dtype

    @property
    def device(self) -> str:
        return str(self._c.device)

    @property
    def itemsize(self) -> int:
        return self._c.itemsize

    @property
    def nbytes(self) -> int:
        return self._c.nbytes

    @property
    def is_contiguous(self) -> bool:
        return self._c.is_contiguous

    @property
    def requires_grad(self) -> bool:
        return self._c.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._c.requires_grad = bool(val)

    def numpy(self) -> np.ndarray:
        """View or copy into NumPy ndarray. Copies to CPU if device is non-CPU."""
        if not self._c.device.is_cpu():
            return np.asarray(self._c.to("cpu"))
        return np.asarray(self._c)

    def item(self) -> float:
        """Extract a scalar value from a 1-element tensor."""
        if self.numel != 1:
            raise ValueError(f"only one element tensors can be converted to Python scalars (got {self.numel})")
        return float(self.numpy().reshape(-1)[0])

    def reshape(self, *shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor._wrap(self._c.reshape(list(shape)), requires_grad=self.requires_grad)

    def view(self, *shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor._wrap(self._c.view(list(shape)), requires_grad=self.requires_grad)

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        return Tensor._wrap(self._c.transpose(dim0, dim1), requires_grad=self.requires_grad)

    @property
    def T(self) -> 'Tensor':
        if self.ndim != 2:
            raise ValueError(f"Tensor.T is only defined for 2D tensors, got {self.ndim}D")
        return self.transpose(0, 1)

    def permute(self, *dims) -> 'Tensor':
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        return Tensor._wrap(self._c.permute(list(dims)), requires_grad=self.requires_grad)

    def contiguous(self) -> 'Tensor':
        return Tensor._wrap(self._c.contiguous(), requires_grad=self.requires_grad)

    def clone(self) -> 'Tensor':
        return Tensor._wrap(self._c.clone(), requires_grad=self.requires_grad)

    def to(self, device_str: str) -> 'Tensor':
        return Tensor._wrap(self._c.to(device_str), requires_grad=self.requires_grad)

    def cpu(self) -> 'Tensor':
        return self.to("cpu")

    def zero_(self) -> 'Tensor':
        self._c.zero_()
        return self

    def fill_(self, value: float) -> 'Tensor':
        self._c.fill_(float(value))
        return self

    def sum(self, dim: int = -1, keepdim: bool = False) -> 'Tensor':
        return Tensor._wrap(self._c.sum(dim, keepdim))

    def mean(self, dim: int = -1, keepdim: bool = False) -> 'Tensor':
        return Tensor._wrap(self._c.mean(dim, keepdim))

    def relu(self) -> 'Tensor':
        return Tensor._wrap(self._c.relu())

    def add(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self.__add__(other)

    def sub(self, other: 'Tensor') -> 'Tensor':
        return self.__sub__(other)

    def mul(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self.__mul__(other)

    def div(self, other: 'Tensor') -> 'Tensor':
        return self.__truediv__(other)

    def matmul(self, other: 'Tensor') -> 'Tensor':
        return self.__matmul__(other)

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        import velocityai.ops as ops
        return ops.add(self, other)

    def __radd__(self, other: Union[float, int]) -> 'Tensor':
        return self.__add__(other)

    def __neg__(self) -> 'Tensor':
        return self * -1.0

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        import velocityai.ops as ops
        return ops.sub(self, other)

    def __rsub__(self, other: Union[float, int]) -> 'Tensor':
        return (-self) + float(other)

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        import velocityai.ops as ops
        return ops.mul(self, other)

    def __rmul__(self, other: Union[float, int]) -> 'Tensor':
        return self.__mul__(other)

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        import velocityai.ops as ops
        return ops.div(self, other)

    def __rtruediv__(self, other: Union[float, int]) -> 'Tensor':
        # Scalar / Tensor
        ones_t = ones(*self.shape, dtype=self.dtype, device=self.device.str)
        return (ones_t * float(other)) / self

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if isinstance(other, Tensor):
            out = Tensor._wrap(self._c.matmul(other._c))
            from velocityai.compiler.tracer import get_active_tracer
            tracer = get_active_tracer()
            if tracer:
                tracer.record_op("MatMul", [self, other], out)
            return out
        return NotImplemented

    def __len__(self) -> int:
        return len(self._c)

    def __getitem__(self, idx) -> 'Tensor':
        import numpy as np
        if isinstance(idx, (int, np.integer)):
            # Slice along dimension 0 with length 1, then squeeze
            if idx < 0:
                idx += self.shape[0]
            sliced = self._c.slice(0, idx, idx + 1, 1)
            # Reshape dropping leading dimension if ndim > 1
            if len(self.shape) > 1:
                return Tensor._wrap(sliced.reshape(list(self.shape[1:])))
            return Tensor._wrap(sliced)
        elif isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.shape[0]
            step = idx.step if idx.step is not None else 1
            return Tensor._wrap(self._c.slice(0, start, stop, step))
        raise NotImplementedError(f"Indexing with {type(idx)} is not yet supported")

    def __repr__(self) -> str:
        return self._c.__repr__()

    def __str__(self) -> str:
        return self._c.__str__()


# Factory functions
def empty(*shape, dtype: DType = float32, device: str = "cpu", requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    dev = Device.from_string(device)
    c_t = _Tensor.empty(list(shape), dtype, dev)
    return Tensor(c_t, requires_grad=requires_grad)

def zeros(*shape, dtype: DType = float32, device: str = "cpu", requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    dev = Device.from_string(device)
    c_t = _Tensor.zeros(list(shape), dtype, dev)
    return Tensor(c_t, requires_grad=requires_grad)

def ones(*shape, dtype: DType = float32, device: str = "cpu", requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    dev = Device.from_string(device)
    c_t = _Tensor.ones(list(shape), dtype, dev)
    return Tensor(c_t, requires_grad=requires_grad)

def full(shape: Sequence[int], fill_value: float, dtype: DType = float32, device: str = "cpu", requires_grad: bool = False) -> Tensor:
    dev = Device.from_string(device)
    c_t = _Tensor.full(list(shape), float(fill_value), dtype, dev)
    return Tensor(c_t, requires_grad=requires_grad)

def randn(*shape, mean: float = 0.0, std: float = 1.0, dtype: DType = float32, device: str = "cpu", requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    dev = Device.from_string(device)
    c_t = _Tensor.randn(list(shape), float(mean), float(std), dtype, dev)
    return Tensor(c_t, requires_grad=requires_grad)

def from_numpy(ndarray: np.ndarray, copy: bool = False, requires_grad: bool = False) -> Tensor:
    """Create a VelocityAI tensor from a NumPy array."""
    if ndarray.dtype not in (np.float32, np.float16, np.int32, np.int8, np.bool_):
        ndarray = ndarray.astype(np.float32)
    c_t = _Tensor.from_numpy(ndarray, copy=copy)
    return Tensor(c_t, requires_grad=requires_grad)

def tensor(data, dtype: DType = float32, device: str = "cpu", requires_grad: bool = False) -> Tensor:
    """Construct a tensor from Python list, scalar, or array."""
    if isinstance(data, Tensor):
        return data.clone()
    arr = np.array(data, dtype=np.float32)
    return from_numpy(arr, copy=True, requires_grad=requires_grad)

__all__ = [
    "Tensor",
    "empty",
    "zeros",
    "ones",
    "full",
    "randn",
    "from_numpy",
    "tensor",
]
