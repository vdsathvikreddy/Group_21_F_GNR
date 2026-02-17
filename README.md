# GNR638 – Assignment 1  
## Custom Deep Learning Framework (Python Backend)

This repository contains a deep learning framework implemented from scratch for GNR638 (Machine Learning for Remote Sensing – II). The framework includes a tensor abstraction with gradient tracking, fully connected layers, convolution layers, activation functions, pooling, loss functions, and optimizers. The training pipeline supports image classification on the provided datasets.

The project supports both a Python backend and an optional C++ backend (via pybind11). The experiments reported here were executed using the Python backend because the C++ extension was not available in the execution environment.

---

## Repository Structure

.
|-- cpp/  
|   |-- CMakeLists.txt  
|   `-- conv_ops.cpp  
|-- framework/  
|   |-- nn/  
|   |   |-- activations.py  
|   |   |-- fast_layers.py  
|   |   |-- layers.py  
|   |   `-- loss.py  
|   |-- optim/  
|   |   `-- optimizer.py  
|   `-- utils/  
|       |-- data.py  
|       |-- parallel_data.py  
|       `-- tensor.py  
|-- build_cpp.sh  
|-- requirements.txt  
|-- train.py  
|-- train_model.py  
`-- README.md  

---

## Requirements

Python 3.12 is recommended.

Create and activate a virtual environment:

    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Minimal dependencies:

    opencv-python
    pybind11  (optional, only if building C++ backend)

---

## Optional: Build C++ Backend

If a compatible compiler is available:

    chmod +x build_cpp.sh
    ./build_cpp.sh

If this step fails, the framework automatically runs in Python-only mode.

---

## Training

Example command:

    python train.py \
        --data_path "/path/to/dataset_parent_folder" \
        --epochs 10 \
        --batch_size 128 \
        --model_type tiny \
        --lr 0.01 \
        --train_split 0.8

Arguments:
- --data_path : Parent folder containing class subfolders
- --epochs : Number of epochs
- --batch_size : Batch size
- --model_type : tiny or ultra
- --lr : Learning rate
- --train_split : Training split ratio
- --mode : train (default) or eval

---

## Evaluation

    python train.py \
        --mode eval \
        --data_path "/path/to/test_parent_folder" \
        --weights model_ultrafast.pkl

---

## Implemented Models

### TinyFastCNN (Used in Experiments)

Input: 32x32x3  

Layers:
1. MaxPool2D (kernel=8, stride=8) → 4x4x3  
2. Flatten → 48  
3. Fully Connected (48 → 32) + ReLU  
4. Fully Connected (32 → num_classes)  

Parameter Counts:
- 10 classes: 1,898 parameters  
- 100 classes: 4,868 parameters  

MACs and FLOPs:
- 10 classes: 1,856 MACs (~3,712 FLOPs)  
- 100 classes: 4,736 MACs (~9,472 FLOPs)  

---

### UltraFastCNN (Alternative)

Input: 32x32x3  

Layers:
1. Conv2D (3 → 8, 3x3, padding=1)  
2. ReLU  
3. MaxPool2D (8x8)  
4. Flatten  
5. Fully Connected (128 → num_classes)  

---

## Experimental Results

### Dataset: data_1 (10 Classes, 60,000 Images)

- Dataset loading time: 52.0217 seconds  
- Images per second: 1153.36  
- Train/Val split: 48,000 / 12,000  
- Model: tiny  
- Epochs: 10  
- Batch size: 128  
- Learning rate: 0.01  
- Average epoch time: 0.86 minutes  
- Total training time: 8.6 minutes  
- Final training loss: 1.2818  
- Final validation loss: 1.2743  
- Final validation accuracy: 56.02%  
- Model saved as: model_ultrafast.pkl  

---

### Dataset: data_2 (100 Classes, 50,000 Images)

- Dataset loading time: 42.1805 seconds  
- Images per second: 1185.38  
- Train/Val split: 40,000 / 10,000  
- Model: tiny  
- Epochs: 10  
- Batch size: 128  
- Learning rate: 0.01  
- Average epoch time: 1.15 minutes  
- Total training time: 11.5 minutes  
- Final training loss: 4.1597  
- Final validation loss: 4.1603  
- Final validation accuracy: 6.86%  
- Model saved as: model_ultrafast.pkl  

---

## Observations

- The tiny model performs reasonably on 10-class classification (56.02% validation accuracy).
- Performance drops significantly on 100-class classification due to extremely limited model capacity.
- Dataset loading and I/O contribute significantly to runtime.
- The Python backend achieved < 2 minutes per epoch for both datasets.

---

## Key Learnings

1. Model capacity must scale with classification complexity.
2. Data loading efficiency significantly affects overall runtime.
3. Even minimal frameworks can successfully train classification models when gradient propagation and tensor design are correctly implemented.

---

## Academic Integrity

AI tools were used only for generating the report and README documentation and for initial guidance on small tensor helper functions. All implementation, debugging, training runs, and experiments were performed independently.

---

23B1042, 23B0906, 23B1010

