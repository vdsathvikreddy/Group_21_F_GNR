"""
Parallel data loading utilities with multiprocessing
"""
import os
import time
import cv2
from multiprocessing import Pool, cpu_count
from framework.tensor import Tensor


def load_single_image(args):
    """Load a single image - used by parallel loader"""
    img_path, target_size, augment = args
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Augmentation
    if augment:
        import random
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        if random.random() > 0.5:
            factor = 0.8 + random.random() * 0.4
            img = (img * factor).clip(0, 255).astype('uint8')
    
    # Normalize
    img = img.astype(float) / 255.0
    
    # Convert to CHW format
    H, W, C = img.shape
    tensor_data = [[[img[h, w, c] for w in range(W)] for h in range(H)] for c in range(C)]
    
    return tensor_data


class ParallelImageDataset:
    """
    Fast parallel image loading using multiprocessing
    """
    def __init__(self, root_dir, target_size=(32, 32), augment=False, num_workers=None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.augment = augment
        self.num_workers = num_workers or cpu_count()
        
        print(f"Loading dataset from {root_dir} with {self.num_workers} workers...")
        start_time = time.time()
        
        # Get class folders
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        # Pre-load all images in parallel
        print(f"Pre-loading {len(self.samples)} images...")
        self.images = []
        self.labels = []
        
        # Prepare arguments for parallel loading
        load_args = [(path, self.target_size, False) for path, _ in self.samples]
        
        # Load in parallel
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(load_single_image, load_args)
        
        # Store loaded images
        for i, result in enumerate(results):
            if result is not None:
                self.images.append(result)
                self.labels.append(self.samples[i][1])
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        print(f"Dataset loading time: {loading_time:.4f} seconds")
        print(f"Loaded {len(self.images)} images across {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
        print(f"Images per second: {len(self.images) / loading_time:.2f}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get pre-loaded image"""
        return self.images[idx], self.labels[idx]


class FastDataLoader:
    """
    Fast data loader with pre-loaded data
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
            
            # Stack into batch tensor
            yield Tensor(batch_images, requires_grad=True), batch_labels
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """
    Split dataset into train and validation sets
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of data for training (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset
    """
    import random
    random.seed(seed)
    
    # Get all indices
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    
    # Split
    split_point = int(len(all_indices) * train_ratio)
    train_indices = all_indices[:split_point]
    val_indices = all_indices[split_point:]
    
    # Create subset datasets
    class SubsetDataset:
        def __init__(self, parent_dataset, indices):
            self.parent = parent_dataset
            self.indices = indices
            self.classes = parent_dataset.classes
            self.class_to_idx = parent_dataset.class_to_idx
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.parent[self.indices[idx]]
    
    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Train ratio: {len(train_dataset) / len(dataset):.2%}")
    
    return train_dataset, val_dataset
