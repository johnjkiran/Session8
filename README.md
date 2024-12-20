# CIFAR10 DNN Implementation

A deep neural network implementation for CIFAR10 classification with the following specifications:
- 4-block architecture (C1, C2, C3, C4)
- Uses Depthwise Separable Convolution
- Uses Dilated Convolution
- Global Average Pooling
- Total RF > 44
- Parameters < 200k

## Requirements 


## Project Structure

- `config/`: Configuration files for the project.
- `models/`: Custom model definitions.
- `utils/`: Utility functions and data loading.
- `train.py`: Main training script.
- `test.py`: Main testing script.
- `README.md`: This file.

## Training

```bash
python train.py
```

## Model Architecture
- C1: Regular Convolution (3 → 24 channels)
- C2: Depthwise Separable Convolution (24 → 48 channels)
- C3: Dilated Depthwise Separable Convolution (48 → 96 channels)
- C4: Strided Depthwise Separable Convolution (96 → 128 channels)
- Global Average Pooling
- Fully Connected Layer (128 → 10 classes)

## Data Augmentation
Uses albumentations library with:
1. Horizontal Flip
2. ShiftScaleRotate
3. CoarseDropout
