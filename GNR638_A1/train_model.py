"""
Ultra-Fast Model - Designed to match friends' 2-3 min/epoch speed
Extreme optimizations for CPU-only training
"""
from framework.nn.module import Module
from framework.nn.fast_layers import FastConv2D, FastLinear, FastReLU, FastMaxPool2D
from framework.nn.layers import Flatten

class UltraFastCNN(Module):
    """
    Ultra-minimal CNN - designed for SPEED
    
    Trade-off: Lower accuracy for much faster training
    Architecture: Single conv + large pooling + tiny FC
    """
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        
        # SINGLE tiny conv layer (8 filters instead of 32)
        self.conv1 = FastConv2D(input_channels, 8, kernel_size=3, padding=1)
        self.register_module('conv1', self.conv1)
        
        self.relu1 = FastReLU()
        self.register_module('relu1', self.relu1)
        
        # LARGE pooling (8x8 instead of 2x2) - massive speedup!
        self.pool1 = FastMaxPool2D(kernel_size=8, stride=8)
        self.register_module('pool1', self.pool1)
        
        # Flatten: 32x32 -> 4x4 after 8x8 pooling
        self.flatten = Flatten()
        self.register_module('flatten', self.flatten)
        
        # Direct to output (no hidden layer!)
        # 8 channels * 4 * 4 = 128 features
        self.fc = FastLinear(8 * 4 * 4, num_classes)
        self.register_module('fc', self.fc)
        
        # Calculate params
        num_params = self._count_params()
        print(f"✅ Ultra-fast model: {num_params:,} parameters (10-20x less than standard)")
    
    def _count_params(self):
        """Count parameters"""
        total = 0
        for param in self.parameters():
            count = 1
            for dim in param.shape:
                count *= dim
            total += count
        return total
    
    def forward(self, x):
        """Super fast forward pass"""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class TinyFastCNN(Module):
    """
    Even smaller - for maximum speed
    NO convolution - just FC layers on downsampled input
    """
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        
        # Direct pooling on input (8x8 pooling)
        self.pool = FastMaxPool2D(kernel_size=8, stride=8)
        self.register_module('pool', self.pool)
        
        # Flatten: 32x32x3 -> 4x4x3 after pooling
        self.flatten = Flatten()
        self.register_module('flatten', self.flatten)
        
        # Small FC layers
        # 4*4*3 = 48 features
        self.fc1 = FastLinear(4 * 4 * input_channels, 32)
        self.register_module('fc1', self.fc1)
        
        self.relu = FastReLU()
        self.register_module('relu', self.relu)
        
        self.fc2 = FastLinear(32, num_classes)
        self.register_module('fc2', self.fc2)
        
        num_params = self._count_params()
        print(f"✅ Tiny model: {num_params:,} parameters (fastest possible)")
    
    def _count_params(self):
        total = 0
        for param in self.parameters():
            count = 1
            for dim in param.shape:
                count *= dim
            total += count
        return total
    
    def forward(self, x):
        """Minimal operations"""
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_ultra_fast_model(num_classes, model_type='ultra'):
    """
    Create ultra-fast model
    
    Args:
        model_type: 'ultra' or 'tiny'
            - 'ultra': Single conv layer (~10K params) - good balance
            - 'tiny': No conv, FC only (~2K params) - maximum speed
    """
    if model_type == 'tiny':
        return TinyFastCNN(num_classes=num_classes)
    else:
        return UltraFastCNN(num_classes=num_classes)
