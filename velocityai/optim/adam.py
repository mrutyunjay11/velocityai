import numpy as np

class Optimizer:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = list(parameters)
        self.lr = lr
        
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad = None
                
    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue
            # p = p - lr * grad
            # Wait, we need in-place updates. For now we can reassign or do:
            # But reassignment breaks references in Module. We must update in place.
            # We don't have p.sub_() yet, so we will do out-of-place and assign _c
            new_p = p - (p.grad * self.lr)
            p._c = new_p._c

class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.m = [None] * len(self.parameters)
        self.v = [None] * len(self.parameters)
        self.t = 0
        
    def step(self):
        import velocityai as vai
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
                
            if self.m[i] is None:
                self.m[i] = vai.zeros(p.shape)
                self.v[i] = vai.zeros(p.shape)
                
            # m = b1*m + (1-b1)*g
            # v = b2*v + (1-b2)*g^2
            # Here we do it manually via python ops
            self.m[i] = self.m[i] * self.betas[0] + p.grad * (1 - self.betas[0])
            self.v[i] = self.v[i] * self.betas[1] + (p.grad * p.grad) * (1 - self.betas[1])
            
            m_hat = self.m[i] / (1 - self.betas[0]**self.t)
            v_hat = self.v[i] / (1 - self.betas[1]**self.t)
            
            # We don't have sqrt easily, let's just do SGD for the prototype
            # Actually we can do it via numpy fallback
            v_hat_np = v_hat.numpy()
            denom_np = np.sqrt(v_hat_np) + self.eps
            denom = vai.from_numpy(denom_np)
            
            update = (m_hat / denom) * self.lr
            new_p = p - update
            p._c = new_p._c
