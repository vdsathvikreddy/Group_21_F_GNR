"""
Optimization algorithms
"""
from framework.tensor import Tensor


class Optimizer:
    """Base optimizer class"""
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr
    
    def step(self):
        """Update parameters - must be implemented by subclasses"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero all parameter gradients"""
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer
    """
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
        
        # Initialize velocity for momentum
        if self.momentum > 0:
            for i, param in enumerate(self.parameters):
                self.velocity[i] = self._zeros_like(param.data)
    
    def step(self):
        """Perform single optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Apply weight decay
            grad = param.grad.data
            if self.weight_decay > 0:
                grad = self._add_weight_decay(grad, param.data, self.weight_decay)
            
            # Apply momentum
            if self.momentum > 0:
                # v = momentum * v + grad
                self.velocity[i] = self._add(
                    self._scalar_mul(self.velocity[i], self.momentum),
                    grad
                )
                update = self.velocity[i]
            else:
                update = grad
            
            # Update parameters: param = param - lr * update
            param.data = self._subtract(param.data, self._scalar_mul(update, self.lr))
    
    @staticmethod
    def _zeros_like(data):
        """Create zeros with same structure"""
        if not isinstance(data, list):
            return 0.0
        return [SGD._zeros_like(item) for item in data]
    
    @staticmethod
    def _scalar_mul(data, scalar):
        """Multiply data by scalar"""
        if not isinstance(data, list):
            return data * scalar
        return [SGD._scalar_mul(item, scalar) for item in data]
    
    @staticmethod
    def _add(data1, data2):
        """Add two data structures"""
        if not isinstance(data1, list):
            return data1 + data2
        return [SGD._add(d1, d2) for d1, d2 in zip(data1, data2)]
    
    @staticmethod
    def _subtract(data1, data2):
        """Subtract data2 from data1"""
        if not isinstance(data1, list):
            return data1 - data2
        return [SGD._subtract(d1, d2) for d1, d2 in zip(data1, data2)]
    
    @staticmethod
    def _add_weight_decay(grad, param, weight_decay):
        """Add weight decay to gradient"""
        if not isinstance(grad, list):
            return grad + weight_decay * param
        return [SGD._add_weight_decay(g, p, weight_decay) 
                for g, p in zip(grad, param)]
    
    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum}, weight_decay={self.weight_decay})"


class Adam(Optimizer):
    """
    Adam optimizer
    """
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # Initialize first and second moment estimates
        self.m = {}
        self.v = {}
        for i, param in enumerate(self.parameters):
            self.m[i] = self._zeros_like(param.data)
            self.v[i] = self._zeros_like(param.data)
    
    def step(self):
        """Perform single optimization step"""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay > 0:
                grad = self._add_weight_decay(grad, param.data, self.weight_decay)
            
            # Update biased first moment estimate
            self.m[i] = self._add(
                self._scalar_mul(self.m[i], self.beta1),
                self._scalar_mul(grad, 1 - self.beta1)
            )
            
            # Update biased second raw moment estimate
            grad_sq = self._square(grad)
            self.v[i] = self._add(
                self._scalar_mul(self.v[i], self.beta2),
                self._scalar_mul(grad_sq, 1 - self.beta2)
            )
            
            # Compute bias-corrected moments
            m_hat = self._scalar_mul(self.m[i], 1.0 / (1.0 - self.beta1 ** self.t))
            v_hat = self._scalar_mul(self.v[i], 1.0 / (1.0 - self.beta2 ** self.t))
            
            # Update parameters
            update = self._div(m_hat, self._add_scalar(self._sqrt(v_hat), self.eps))
            param.data = self._subtract(param.data, self._scalar_mul(update, self.lr))
    
    @staticmethod
    def _zeros_like(data):
        """Create zeros with same structure"""
        if not isinstance(data, list):
            return 0.0
        return [Adam._zeros_like(item) for item in data]
    
    @staticmethod
    def _scalar_mul(data, scalar):
        """Multiply data by scalar"""
        if not isinstance(data, list):
            return data * scalar
        return [Adam._scalar_mul(item, scalar) for item in data]
    
    @staticmethod
    def _add(data1, data2):
        """Add two data structures"""
        if not isinstance(data1, list):
            return data1 + data2
        return [Adam._add(d1, d2) for d1, d2 in zip(data1, data2)]
    
    @staticmethod
    def _subtract(data1, data2):
        """Subtract data2 from data1"""
        if not isinstance(data1, list):
            return data1 - data2
        return [Adam._subtract(d1, d2) for d1, d2 in zip(data1, data2)]
    
    @staticmethod
    def _square(data):
        """Square each element"""
        if not isinstance(data, list):
            return data * data
        return [Adam._square(item) for item in data]
    
    @staticmethod
    def _sqrt(data):
        """Square root of each element"""
        import math
        if not isinstance(data, list):
            return math.sqrt(data)
        return [Adam._sqrt(item) for item in data]
    
    @staticmethod
    def _div(data1, data2):
        """Element-wise division"""
        if not isinstance(data1, list):
            return data1 / data2
        return [Adam._div(d1, d2) for d1, d2 in zip(data1, data2)]
    
    @staticmethod
    def _add_scalar(data, scalar):
        """Add scalar to each element"""
        if not isinstance(data, list):
            return data + scalar
        return [Adam._add_scalar(item, scalar) for item in data]
    
    @staticmethod
    def _add_weight_decay(grad, param, weight_decay):
        """Add weight decay to gradient"""
        if not isinstance(grad, list):
            return grad + weight_decay * param
        return [Adam._add_weight_decay(g, p, weight_decay) 
                for g, p in zip(grad, param)]
    
    def __repr__(self):
        return f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), eps={self.eps})"
