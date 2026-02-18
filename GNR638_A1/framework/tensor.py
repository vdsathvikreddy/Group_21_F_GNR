"""
Tensor class with automatic differentiation support
"""
import math
from typing import Optional, Tuple, Union, List


class Tensor:
    """
    Core tensor class with autograd support
    """
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if isinstance(data, list):
            self.data = self._list_to_nested(data)
        else:
            self.data = data
        
        self.shape = self._get_shape(self.data)
        self.requires_grad = requires_grad
        self.grad = None
        
        # For autograd
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    @staticmethod
    def _list_to_nested(lst):
        """Convert nested list to data structure"""
        if not isinstance(lst, list):
            return lst
        return [Tensor._list_to_nested(item) if isinstance(item, list) else item for item in lst]
    
    @staticmethod
    def _get_shape(data):
        """Get shape of nested list structure"""
        if not isinstance(data, list):
            return ()
        shape = [len(data)]
        if data and isinstance(data[0], list):
            shape.extend(Tensor._get_shape(data[0]))
        return tuple(shape)
    
    def zeros(self, shape):
        """Create tensor filled with zeros"""
        return Tensor(self._create_nested(shape, 0.0))
    
    def ones(self, shape):
        """Create tensor filled with ones"""
        return Tensor(self._create_nested(shape, 1.0))
    
    def randn(self, shape, mean=0.0, std=1.0):
        """Create tensor with random normal values"""
        import random
        def _randn():
            # Box-Muller transform
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            return mean + z * std
        
        return Tensor(self._create_nested(shape, None, _randn))
    
    @staticmethod
    def _create_nested(shape, value, func=None):
        """Create nested list structure with given shape"""
        if len(shape) == 0:
            return func() if func else value
        if len(shape) == 1:
            return [func() if func else value for _ in range(shape[0])]
        return [Tensor._create_nested(shape[1:], value, func) for _ in range(shape[0])]
    
    def __add__(self, other):
        """Element-wise addition"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._add_data(self.data, other.data), 
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros_like(self)
                self.grad.data = self._add_data(self.grad.data, out.grad.data)
            if other.requires_grad:
                if other.grad is None:
                    other.grad = Tensor.zeros_like(other)
                other.grad.data = self._add_data(other.grad.data, out.grad.data)
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """Element-wise multiplication"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._mul_data(self.data, other.data),
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros_like(self)
                grad_contrib = self._mul_data(other.data, out.grad.data)
                self.grad.data = self._add_data(self.grad.data, grad_contrib)
            if other.requires_grad:
                if other.grad is None:
                    other.grad = Tensor.zeros_like(other)
                grad_contrib = self._mul_data(self.data, out.grad.data)
                other.grad.data = self._add_data(other.grad.data, grad_contrib)
        
        out._backward = _backward
        return out
    
    @staticmethod
    def _add_data(a, b):
        """Recursive addition"""
        if not isinstance(a, list):
            return a + b
        return [Tensor._add_data(ai, bi) for ai, bi in zip(a, b)]
    
    @staticmethod
    def _mul_data(a, b):
        """Recursive multiplication"""
        if not isinstance(a, list):
            return a * b
        return [Tensor._mul_data(ai, bi) for ai, bi in zip(a, b)]
    
    @staticmethod
    def zeros_like(tensor):
        """Create zero tensor with same shape"""
        return Tensor(Tensor._create_nested(tensor.shape, 0.0))
    
    def backward(self):
        """Compute gradients via backpropagation"""
        # Topological sort
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient
        self.grad = Tensor(self._create_nested(self.shape, 1.0))
        
        # Backpropagate
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Zero out gradients"""
        self.grad = None
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
    def item(self):
        """Get scalar value"""
        if self.shape == ():
            return self.data
        raise ValueError("Only single element tensors can be converted to Python scalars")
    
    def numpy(self):
        """Convert to nested list (since we can't use numpy)"""
        return self.data
