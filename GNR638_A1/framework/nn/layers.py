"""
Neural network layers
"""
import math
from framework.tensor import Tensor
from framework.nn.module import Module


class Linear(Module):
    """
    Fully connected (linear) layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with He initialization
        std = math.sqrt(2.0 / in_features)
        self.weight = Tensor(self._randn((out_features, in_features), 0, std), requires_grad=True)
        self.register_parameter('weight', self.weight)
        
        if bias:
            self.bias = Tensor(self._zeros((out_features,)), requires_grad=True)
            self.register_parameter('bias', self.bias)
        else:
            self.bias = None
    
    @staticmethod
    def _randn(shape, mean=0, std=1):
        """Create random normal initialized nested list"""
        import random
        def rand_val():
            u1, u2 = random.random(), random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            return mean + z * std
        
        if len(shape) == 1:
            return [rand_val() for _ in range(shape[0])]
        return [[rand_val() for _ in range(shape[1])] for _ in range(shape[0])]
    
    @staticmethod
    def _zeros(shape):
        """Create zero-initialized nested list"""
        if len(shape) == 1:
            return [0.0 for _ in range(shape[0])]
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
    
    def forward(self, x):
        """
        Forward pass: y = xW^T + b
        x: (batch_size, in_features)
        output: (batch_size, out_features)
        """
        # Matrix multiplication: x @ W.T
        batch_size = x.shape[0]
        output_data = []
        
        for b in range(batch_size):
            row = []
            for i in range(self.out_features):
                val = 0.0
                for j in range(self.in_features):
                    val += x.data[b][j] * self.weight.data[i][j]
                if self.bias is not None:
                    val += self.bias.data[i]
                row.append(val)
            output_data.append(row)
        
        out = Tensor(output_data, requires_grad=x.requires_grad or self.weight.requires_grad)
        
        # Store for backward pass
        out._prev = {x, self.weight}
        if self.bias is not None:
            out._prev.add(self.bias)
        
        def _backward():
            # Gradient w.r.t input
            if x.requires_grad:
                if x.grad is None:
                    x.grad = Tensor(self._zeros(x.shape))
                for b in range(batch_size):
                    for j in range(self.in_features):
                        for i in range(self.out_features):
                            x.grad.data[b][j] += out.grad.data[b][i] * self.weight.data[i][j]
            
            # Gradient w.r.t weight
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = Tensor(self._zeros(self.weight.shape))
                for i in range(self.out_features):
                    for j in range(self.in_features):
                        for b in range(batch_size):
                            self.weight.grad.data[i][j] += out.grad.data[b][i] * x.data[b][j]
            
            # Gradient w.r.t bias
            if self.bias is not None and self.bias.requires_grad:
                if self.bias.grad is None:
                    self.bias.grad = Tensor(self._zeros(self.bias.shape))
                for i in range(self.out_features):
                    for b in range(batch_size):
                        self.bias.grad.data[i] += out.grad.data[b][i]
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"


class Conv2D(Module):
    """
    2D Convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights
        k = 1.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        std = math.sqrt(k)
        
        # Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        self.weight = Tensor(
            self._randn((out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]), 0, std),
            requires_grad=True
        )
        self.register_parameter('weight', self.weight)
        
        if bias:
            self.bias = Tensor(self._zeros((out_channels,)), requires_grad=True)
            self.register_parameter('bias', self.bias)
        else:
            self.bias = None
    
    @staticmethod
    def _randn(shape, mean=0, std=1):
        """4D random initialization"""
        import random
        def rand_val():
            u1, u2 = random.random(), random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            return mean + z * std
        
        return [[[[rand_val() for _ in range(shape[3])] 
                  for _ in range(shape[2])] 
                 for _ in range(shape[1])] 
                for _ in range(shape[0])]
    
    @staticmethod
    def _zeros(shape):
        """Zero initialization"""
        if len(shape) == 1:
            return [0.0 for _ in range(shape[0])]
        return [[[[0.0 for _ in range(shape[3])] 
                  for _ in range(shape[2])] 
                 for _ in range(shape[1])] 
                for _ in range(shape[0])]
    
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, in_channels, height, width)
        output: (batch_size, out_channels, out_height, out_width)
        """
        batch_size, _, H, W = x.shape
        
        # Add padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = self._add_padding(x.data, self.padding)
            H += 2 * self.padding[0]
            W += 2 * self.padding[1]
        else:
            x_padded = x.data
        
        # Calculate output dimensions
        out_h = (H - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (W - self.kernel_size[1]) // self.stride[1] + 1
        
        # Perform convolution
        output_data = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] 
                        for _ in range(self.out_channels)] 
                       for _ in range(batch_size)]
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        val = 0.0
                        h_start = oh * self.stride[0]
                        w_start = ow * self.stride[1]
                        
                        for ic in range(self.in_channels):
                            for kh in range(self.kernel_size[0]):
                                for kw in range(self.kernel_size[1]):
                                    h_idx = h_start + kh
                                    w_idx = w_start + kw
                                    val += (x_padded[b][ic][h_idx][w_idx] * 
                                           self.weight.data[oc][ic][kh][kw])
                        
                        if self.bias is not None:
                            val += self.bias.data[oc]
                        output_data[b][oc][oh][ow] = val
        
        out = Tensor(output_data, requires_grad=x.requires_grad or self.weight.requires_grad)
        # Note: Backward pass implementation needed for training
        return out
    
    @staticmethod
    def _add_padding(data, padding):
        """Add zero padding to input"""
        batch_size = len(data)
        channels = len(data[0])
        H = len(data[0][0])
        W = len(data[0][0][0])
        
        padded = [[[[0.0 for _ in range(W + 2*padding[1])] 
                    for _ in range(H + 2*padding[0])] 
                   for _ in range(channels)] 
                  for _ in range(batch_size)]
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(H):
                    for w in range(W):
                        padded[b][c][h + padding[0]][w + padding[1]] = data[b][c][h][w]
        
        return padded
    
    def __repr__(self):
        return f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size})"


class MaxPool2D(Module):
    """
    2D Max Pooling layer
    """
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if not isinstance(self.stride, tuple):
            self.stride = (self.stride, self.stride)
    
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, channels, height, width)
        """
        batch_size, channels, H, W = x.shape
        
        out_h = (H - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (W - self.kernel_size[1]) // self.stride[1] + 1
        
        output_data = [[[[0.0 for _ in range(out_w)] for _ in range(out_h)] 
                        for _ in range(channels)] 
                       for _ in range(batch_size)]
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride[0]
                        w_start = ow * self.stride[1]
                        
                        max_val = float('-inf')
                        for kh in range(self.kernel_size[0]):
                            for kw in range(self.kernel_size[1]):
                                val = x.data[b][c][h_start + kh][w_start + kw]
                                if val > max_val:
                                    max_val = val
                        
                        output_data[b][c][oh][ow] = max_val
        
        return Tensor(output_data, requires_grad=x.requires_grad)
    
    def __repr__(self):
        return f"MaxPool2D(kernel_size={self.kernel_size}, stride={self.stride})"


class Flatten(Module):
    """
    Flatten layer to convert 4D tensor to 2D
    """
    def forward(self, x):
        """
        x: (batch_size, channels, height, width)
        output: (batch_size, channels * height * width)
        """
        batch_size = x.shape[0]
        flattened_size = 1
        for dim in x.shape[1:]:
            flattened_size *= dim
        
        output_data = []
        for b in range(batch_size):
            row = []
            for c in range(x.shape[1]):
                for h in range(x.shape[2]):
                    for w in range(x.shape[3]):
                        row.append(x.data[b][c][h][w])
            output_data.append(row)
        
        return Tensor(output_data, requires_grad=x.requires_grad)
    
    def __repr__(self):
        return "Flatten()"
