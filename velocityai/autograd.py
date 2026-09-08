import velocityai as vai
from velocityai.tensor import Tensor
from collections import defaultdict
import numpy as np

# Monkey patch Tensor to support autograd fields
if not hasattr(Tensor, 'grad'):
    Tensor.grad = None
if not hasattr(Tensor, 'grad_fn'):
    Tensor.grad_fn = None
if not hasattr(Tensor, 'requires_grad'):
    # Pybind11 exposes a property requires_grad, but we can override in python
    pass

class Context:
    def __init__(self):
        self.saved_tensors = ()
        
    def save_for_backward(self, *args):
        self.saved_tensors = args

class Function:
    @classmethod
    def apply(cls, *args):
        ctx = Context()
        
        # Strip Tensors to pass to forward if needed, or pass directly
        result = cls.forward(ctx, *args)
        
        # Check if any input requires grad
        requires_grad = any(isinstance(arg, Tensor) and getattr(arg, 'requires_grad', False) for arg in args)
        
        if requires_grad and isinstance(result, Tensor):
            result.requires_grad = True
            result.grad_fn = (cls, ctx, args)
            
        return result
        
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError
        
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

def _backward(tensor, grad=None):
    if not getattr(tensor, 'requires_grad', False):
        raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        
    if grad is None:
        if tensor.numel == 1:
            grad = vai.ones(tensor.shape)
        else:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
            
    # Topological sort
    topo = []
    visited = set()
    def build_topo(t):
        if t not in visited and hasattr(t, 'grad_fn') and t.grad_fn is not None:
            visited.add(t)
            cls, ctx, args = t.grad_fn
            for arg in args:
                if isinstance(arg, Tensor):
                    build_topo(arg)
            topo.append(t)
            
    build_topo(tensor)
    
    tensor.grad = grad
    
    for t in reversed(topo):
        cls, ctx, args = t.grad_fn
        grads = cls.backward(ctx, t.grad)
        
        if not isinstance(grads, tuple):
            grads = (grads,)
            
        # Accumulate gradients
        tensor_args = [arg for arg in args if isinstance(arg, Tensor)]
        
        for arg, g in zip(args, grads):
            if isinstance(arg, Tensor) and arg.requires_grad and g is not None:
                if arg.grad is None:
                    arg.grad = g
                else:
                    arg.grad = arg.grad + g

# Attach backward to Tensor
Tensor.backward = _backward

# Override ops with Autograd support
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        # Using raw op to prevent infinite recursion
        return vai.Tensor._wrap(a._c.add(b._c)) if isinstance(b, Tensor) else vai.Tensor._wrap(a._c.add_scalar(float(b)))
        
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        if isinstance(b, Tensor):
            return grad_output, grad_output
        return grad_output, None

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        return vai.Tensor._wrap(a._c.sub(b._c)) if isinstance(b, Tensor) else vai.Tensor._wrap(a._c.sub(float(b)))
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output * -1.0

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        if isinstance(b, Tensor):
            return vai.Tensor._wrap(a._c.mul(b._c))
        else:
            return vai.Tensor._wrap(a._c.mul_scalar(float(b)))
            
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        if isinstance(b, Tensor):
            # We don't have basic tensor multiplication elementwise in python wrappers easily mapped, wait we do!
            # It's vai.mul
            return vai.mul(grad_output, b), vai.mul(grad_output, a)
        return vai.mul(grad_output, b), None

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return vai.Tensor._wrap(a._c.matmul(b._c))
        
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # dL/dA = dL/dY @ B.T
        # dL/dB = A.T @ dL/dY
        
        # We need transpose. 
        a_T = a.transpose(0, 1)
        b_T = b.transpose(0, 1)
        
        grad_a = vai.matmul(grad_output, b_T)
        grad_b = vai.matmul(a_T, grad_output)
        
        return grad_a, grad_b

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return vai.Tensor._wrap(a._c.relu())
        
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        # gradient is grad_output * (a > 0)
        # Hack for a > 0 using numpy since we don't have comparison ops yet in C++
        a_np = a.numpy()
        mask = (a_np > 0).astype(np.float32)
        mask_t = vai.from_numpy(mask)
        return vai.mul(grad_output, mask_t)

class Sum(Function):
    @staticmethod
    def forward(ctx, a):
        # We assume full reduction for now
        ctx.save_for_backward(a)
        return vai.Tensor._wrap(a._c.sum(-1, False))
        
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        # gradient of sum is just 1s of shape a, multiplied by grad_output
        grad = vai.ones(a.shape)
        # broadcast multiply
        # Wait, grad_output is scalar.
        val = grad_output.numpy()[0]
        return vai.mul(grad, val)

class CrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits, targets):
        ctx.save_for_backward(logits, targets)
        # Compute in numpy for now as cross_entropy kernel was a stub
        logits_np = logits.numpy()
        targets_np = targets.numpy().astype(int)
        
        # Softmax
        exps = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        # NLL
        batch = logits_np.shape[0]
        loss = 0.0
        for i in range(batch):
            loss -= np.log(probs[i, targets_np[i]] + 1e-7)
            
        return vai.tensor([loss / batch], dtype=vai.float32)
        
    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        logits_np = logits.numpy()
        targets_np = targets.numpy().astype(int)
        
        exps = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        batch = logits_np.shape[0]
        grad_logits = probs.copy()
        for i in range(batch):
            grad_logits[i, targets_np[i]] -= 1.0
            
        grad_logits /= batch
        
        grad_out_np = grad_output.numpy()[0]
        grad_logits *= grad_out_np
        
        return vai.from_numpy(grad_logits.astype(np.float32)), None
