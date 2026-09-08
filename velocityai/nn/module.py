class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True
        
    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif hasattr(value, 'requires_grad'):
            self._parameters[name] = value
        super().__setattr__(name, value)
        
    def parameters(self):
        for name, param in self._parameters.items():
            yield param
        for name, module in self._modules.items():
            for param in module.parameters():
                yield param
                
    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
            
    def eval(self):
        self.train(False)
        
    def to(self, device: str):
        """Moves all parameters to the given device."""
        for name, param in self._parameters.items():
            if param is not None:
                try:
                    new_param = param.to(device)
                    new_param.requires_grad = param.requires_grad
                    self._parameters[name] = new_param
                    super().__setattr__(name, new_param)
                except AttributeError as e:
                    print(f"Error moving parameter '{name}' of type {type(param)}")
                    raise e
        for module in self._modules.values():
            module.to(device)
        # Also move items in lists if they are modules
        for attr_name in dir(self):
            try:
                attr = getattr(self, attr_name)
                if isinstance(attr, list):
                    for item in attr:
                        if isinstance(item, Module):
                            item.to(device)
            except:
                pass
        return self
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError
