from velocityai.nn.module import Module
import velocityai as vai
import velocityai.autograd as ag
import math
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, init_weights=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if init_weights:
            # Native weight layout: (out_features, in_features)
            std = math.sqrt(1.0 / in_features)
            self.weight = vai.randn(out_features, in_features) * std
            self.weight.requires_grad = False
            
            if bias:
                self.bias = vai.randn(1, out_features) * std
                self.bias.requires_grad = False
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None
            
    def forward(self, x):
        from _velocityai_c import linear
        out = vai.Tensor._wrap(linear(x._c, self.weight._c))
            
        if self.bias is not None:
            out = out + self.bias
        return out

class BroadcastBias(ag.Function):
    @staticmethod
    def backward(ctx, grad_output):
        # sum across batch dimension to get (1, out_features)
        grad_np = np.sum(grad_output.numpy(), axis=0, keepdims=True)
        return vai.from_numpy(grad_np)
