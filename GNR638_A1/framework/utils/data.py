"""
Data loading utilities
"""
import os
import time
import cv2
from framework.tensor import Tensor


class ImageFolderDataset:
    """
    Load images from folder structure where each subfolder is a class
    """
    def __init__(self, root_dir, target_size=(32, 32), augment=False):
        self.root_dir = root_dir
        self.target_size = target_size
        self.augment = augment
        
        print(f"Loading dataset from {root_dir}...")
        start_time = time.time()
        
        # Get all class folders
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        print(f"Dataset loading time: {loading_time:.4f} seconds")
        print(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single image and label"""
        img_path, label = self.samples[idx]
        
        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, self.target_size)
        
        # Apply augmentation if enabled
        if self.augment:
            img = self._augment(img)
        
        # Normalize to [0, 1]
        img = img.astype(float) / 255.0
        
        # Convert to tensor format (C, H, W)
        img_tensor = self._to_tensor(img)
        
        return img_tensor, label
    
    def _augment(self, img):
        """Apply simple data augmentation"""
        import random
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # Random brightness adjustment
        if random.random() > 0.5:
            factor = 0.8 + random.random() * 0.4  # 0.8 to 1.2
            img = (img * factor).clip(0, 255).astype('uint8')
        
        return img
    
    @staticmethod
    def _to_tensor(img):
        """
        Convert HWC image to CHW tensor format
        img: numpy array (H, W, C)
        returns: nested list (C, H, W)
        """
        H, W, C = img.shape
        tensor_data = [[[img[h, w, c] for w in range(W)] for h in range(H)] for c in range(C)]
        return tensor_data


class DataLoader:
    """
    Data loader for batching
    """
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                img, label = self.dataset[idx]
                batch_images.append(img)
                batch_labels.append(label)
            
            # Stack images into batch tensor (batch_size, C, H, W)
            yield Tensor(batch_images, requires_grad=True), batch_labels
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def count_parameters(model):
    """Count trainable parameters in model"""
    total = 0
    for param in model.parameters():
        count = 1
        for dim in param.shape:
            count *= dim
        total += count
    return total


def calculate_macs_flops(model, input_shape):
    """
    Calculate MACs and FLOPs for the model
    This is a simplified version - you'll need to extend based on your architecture
    """
    # For Conv2D: MACs = output_h * output_w * kernel_h * kernel_w * in_channels * out_channels
    # FLOPs ≈ 2 * MACs (multiply-add counted as 2 operations)
    
    macs = 0
    flops = 0
    
    # You'll need to traverse the model and calculate based on layer types
    # This is a placeholder - implement based on your specific architecture
    
    return macs, flops
