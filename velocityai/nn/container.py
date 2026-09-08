from velocityai.nn.module import Module

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = args
        for i, module in enumerate(args):
            self._modules[str(i)] = module
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
