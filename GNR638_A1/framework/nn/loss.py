"""
Loss functions
"""
import math
from framework.tensor import Tensor
from framework.nn.module import Module


class CrossEntropyLoss(Module):
    """
    Cross Entropy Loss for classification
    Combines LogSoftmax and NLLLoss
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        """
        predictions: (batch_size, num_classes) - raw logits
        targets: (batch_size,) - class indices
        """
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]
        
        # Convert targets to list if it's a Tensor
        if isinstance(targets, Tensor):
            target_data = targets.data
        else:
            target_data = targets
        
        total_loss = 0.0
        
        for b in range(batch_size):
            # Find max for numerical stability
            max_val = max(predictions.data[b])
            
            # Compute log-sum-exp
            sum_exp = 0.0
            for i in range(num_classes):
                sum_exp += math.exp(predictions.data[b][i] - max_val)
            
            log_sum_exp = max_val + math.log(sum_exp)
            
            # Cross entropy loss for this sample
            target_class = int(target_data[b])
            loss = log_sum_exp - predictions.data[b][target_class]
            total_loss += loss
        
        # Average over batch
        loss_val = total_loss / batch_size
        
        out = Tensor(loss_val, requires_grad=predictions.requires_grad)
        
        def _backward():
            if predictions.requires_grad:
                if predictions.grad is None:
                    predictions.grad = Tensor([[0.0 for _ in range(num_classes)] 
                                              for _ in range(batch_size)])
                
                # Gradient of cross entropy: softmax - one_hot
                for b in range(batch_size):
                    # Compute softmax
                    max_val = max(predictions.data[b])
                    exp_vals = [math.exp(predictions.data[b][i] - max_val) 
                               for i in range(num_classes)]
                    sum_exp = sum(exp_vals)
                    softmax = [exp_vals[i] / sum_exp for i in range(num_classes)]
                    
                    # Gradient
                    target_class = int(target_data[b])
                    for i in range(num_classes):
                        grad = softmax[i]
                        if i == target_class:
                            grad -= 1.0
                        predictions.grad.data[b][i] += grad / batch_size
        
        out._prev = {predictions}
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "CrossEntropyLoss()"


class MSELoss(Module):
    """
    Mean Squared Error Loss
    """
    def forward(self, predictions, targets):
        """
        predictions: predicted values
        targets: ground truth values
        """
        # Flatten if needed
        pred_flat = self._flatten(predictions.data)
        target_flat = self._flatten(targets.data if isinstance(targets, Tensor) else targets)
        
        n = len(pred_flat)
        total_loss = 0.0
        
        for i in range(n):
            diff = pred_flat[i] - target_flat[i]
            total_loss += diff * diff
        
        loss_val = total_loss / n
        out = Tensor(loss_val, requires_grad=predictions.requires_grad)
        
        def _backward():
            if predictions.requires_grad:
                if predictions.grad is None:
                    predictions.grad = Tensor(self._zeros_like(predictions.data))
                
                # Gradient: 2(pred - target) / n
                self._mse_backward(predictions.data, target_flat, predictions.grad.data, n)
        
        out._prev = {predictions}
        out._backward = _backward
        return out
    
    @staticmethod
    def _flatten(data):
        """Flatten nested list"""
        if not isinstance(data, list):
            return [data]
        result = []
        for item in data:
            if isinstance(item, list):
                result.extend(MSELoss._flatten(item))
            else:
                result.append(item)
        return result
    
    @staticmethod
    def _zeros_like(data):
        """Create zeros with same structure"""
        if not isinstance(data, list):
            return 0.0
        return [MSELoss._zeros_like(item) for item in data]
    
    @staticmethod
    def _mse_backward(pred_data, target_flat, grad_data, n, idx=[0]):
        """Backward pass for MSE"""
        if not isinstance(pred_data, list):
            grad_data += 2.0 * (pred_data - target_flat[idx[0]]) / n
            idx[0] += 1
            return
        
        for i in range(len(pred_data)):
            MSELoss._mse_backward(pred_data[i], target_flat, grad_data[i], n, idx)
    
    def __repr__(self):
        return "MSELoss()"
