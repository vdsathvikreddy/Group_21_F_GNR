"""
Activation functions
"""
import math
from framework.tensor import Tensor
from framework.nn.module import Module


class ReLU(Module):
    """
    Rectified Linear Unit activation
    """
    def forward(self, x):
        """Apply ReLU: max(0, x)"""
        output_data = self._apply_relu(x.data)
        out = Tensor(output_data, requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = Tensor(self._zeros_like(x.data))
                # Gradient is 1 where x > 0, else 0
                self._relu_backward(x.data, out.grad.data, x.grad.data)
        
        out._prev = {x}
        out._backward = _backward
        return out
    
    @staticmethod
    def _apply_relu(data):
        """Recursively apply ReLU"""
        if not isinstance(data, list):
            return max(0.0, data)
        return [ReLU._apply_relu(item) for item in data]
    
    @staticmethod
    def _zeros_like(data):
        """Create zeros with same structure"""
        if not isinstance(data, list):
            return 0.0
        return [ReLU._zeros_like(item) for item in data]
    
    @staticmethod
    def _relu_backward(x_data, grad_output, grad_input):
        """Backward pass for ReLU"""
        if not isinstance(x_data, list):
            return
        for i in range(len(x_data)):
            if isinstance(x_data[i], list):
                ReLU._relu_backward(x_data[i], grad_output[i], grad_input[i])
            else:
                grad_input[i] += grad_output[i] if x_data[i] > 0 else 0.0
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """
    Sigmoid activation
    """
    def forward(self, x):
        """Apply sigmoid: 1 / (1 + exp(-x))"""
        output_data = self._apply_sigmoid(x.data)
        out = Tensor(output_data, requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = Tensor(self._zeros_like(x.data))
                # Gradient: sigmoid(x) * (1 - sigmoid(x))
                self._sigmoid_backward(output_data, out.grad.data, x.grad.data)
        
        out._prev = {x}
        out._backward = _backward
        return out
    
    @staticmethod
    def _apply_sigmoid(data):
        """Recursively apply sigmoid"""
        if not isinstance(data, list):
            return 1.0 / (1.0 + math.exp(-data))
        return [Sigmoid._apply_sigmoid(item) for item in data]
    
    @staticmethod
    def _zeros_like(data):
        """Create zeros with same structure"""
        if not isinstance(data, list):
            return 0.0
        return [Sigmoid._zeros_like(item) for item in data]
    
    @staticmethod
    def _sigmoid_backward(sig_output, grad_output, grad_input):
        """Backward pass for sigmoid"""
        if not isinstance(sig_output, list):
            return
        for i in range(len(sig_output)):
            if isinstance(sig_output[i], list):
                Sigmoid._sigmoid_backward(sig_output[i], grad_output[i], grad_input[i])
            else:
                grad_input[i] += grad_output[i] * sig_output[i] * (1.0 - sig_output[i])
    
    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """
    Hyperbolic tangent activation
    """
    def forward(self, x):
        """Apply tanh"""
        output_data = self._apply_tanh(x.data)
        out = Tensor(output_data, requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = Tensor(self._zeros_like(x.data))
                # Gradient: 1 - tanh(x)^2
                self._tanh_backward(output_data, out.grad.data, x.grad.data)
        
        out._prev = {x}
        out._backward = _backward
        return out
    
    @staticmethod
    def _apply_tanh(data):
        """Recursively apply tanh"""
        if not isinstance(data, list):
            return math.tanh(data)
        return [Tanh._apply_tanh(item) for item in data]
    
    @staticmethod
    def _zeros_like(data):
        """Create zeros with same structure"""
        if not isinstance(data, list):
            return 0.0
        return [Tanh._zeros_like(item) for item in data]
    
    @staticmethod
    def _tanh_backward(tanh_output, grad_output, grad_input):
        """Backward pass for tanh"""
        if not isinstance(tanh_output, list):
            return
        for i in range(len(tanh_output)):
            if isinstance(tanh_output[i], list):
                Tanh._tanh_backward(tanh_output[i], grad_output[i], grad_input[i])
            else:
                grad_input[i] += grad_output[i] * (1.0 - tanh_output[i] ** 2)
    
    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    """
    Softmax activation for multi-class classification
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        """
        Apply softmax along specified dimension
        For numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        """
        # Assuming x is (batch_size, num_classes)
        batch_size = x.shape[0]
        num_classes = x.shape[1]
        
        output_data = []
        for b in range(batch_size):
            # Find max for numerical stability
            max_val = max(x.data[b])
            
            # Compute exp(x - max)
            exp_vals = [math.exp(x.data[b][i] - max_val) for i in range(num_classes)]
            
            # Compute sum
            sum_exp = sum(exp_vals)
            
            # Normalize
            row = [exp_vals[i] / sum_exp for i in range(num_classes)]
            output_data.append(row)
        
        return Tensor(output_data, requires_grad=x.requires_grad)
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"
