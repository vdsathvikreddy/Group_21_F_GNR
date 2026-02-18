"""
Base Module class for all neural network layers
"""
from typing import Iterator, Tuple
from framework.tensor import Tensor


class Module:
    """
    Base class for all neural network modules
    """
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs):
        """Make module callable"""
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> Iterator[Tensor]:
        """Return iterator over module parameters"""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Return iterator over module parameters with names"""
        for name, param in self._parameters.items():
            yield name, param
        for name, module in self._modules.items():
            for subname, param in module.named_parameters():
                yield f"{name}.{subname}", param
    
    def register_parameter(self, name: str, param: Tensor):
        """Register a parameter"""
        self._parameters[name] = param
    
    def register_module(self, name: str, module: 'Module'):
        """Register a submodule"""
        self._modules[name] = module
    
    def train(self, mode: bool = True):
        """Set training mode"""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
    
    def eval(self):
        """Set evaluation mode"""
        self.train(False)
    
    def zero_grad(self):
        """Zero gradients for all parameters"""
        for param in self.parameters():
            param.zero_grad()
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Sequential(Module):
    """
    Sequential container for modules
    """
    def __init__(self, *modules):
        super().__init__()
        for idx, module in enumerate(modules):
            self.register_module(str(idx), module)
        self.module_list = list(modules)
    
    def forward(self, x):
        """Forward through all modules sequentially"""
        for module in self.module_list:
            x = module(x)
        return x
    
    def __repr__(self):
        return f"Sequential({', '.join([str(m) for m in self.module_list])})"
