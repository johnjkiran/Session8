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

## Output from Torch Summary
Model Summary:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 24, 32, 32]             672
       BatchNorm2d-2           [-1, 24, 32, 32]              48
              ReLU-3           [-1, 24, 32, 32]               0
            Conv2d-4           [-1, 24, 32, 32]             240
            Conv2d-5           [-1, 48, 32, 32]           1,200
DepthwiseSeparableConv-6           [-1, 48, 32, 32]               0
       BatchNorm2d-7           [-1, 48, 32, 32]              96
              ReLU-8           [-1, 48, 32, 32]               0
            Conv2d-9           [-1, 48, 32, 32]             480
           Conv2d-10           [-1, 96, 32, 32]           4,704
DepthwiseSeparableConv-11           [-1, 96, 32, 32]               0
            Conv2d-9           [-1, 48, 32, 32]             480
           Conv2d-10           [-1, 96, 32, 32]           4,704
DepthwiseSeparableConv-11           [-1, 96, 32, 32]               0
DepthwiseSeparableConv-11           [-1, 96, 32, 32]               0
      BatchNorm2d-12           [-1, 96, 32, 32]             192
             ReLU-13           [-1, 96, 32, 32]               0
           Conv2d-14           [-1, 96, 16, 16]             960
      BatchNorm2d-12           [-1, 96, 32, 32]             192
             ReLU-13           [-1, 96, 32, 32]               0
           Conv2d-14           [-1, 96, 16, 16]             960
           Conv2d-14           [-1, 96, 16, 16]             960
           Conv2d-15          [-1, 128, 16, 16]          12,416
DepthwiseSeparableConv-16          [-1, 128, 16, 16]               0
           Conv2d-15          [-1, 128, 16, 16]          12,416
DepthwiseSeparableConv-16          [-1, 128, 16, 16]               0
      BatchNorm2d-17          [-1, 128, 16, 16]             256
DepthwiseSeparableConv-16          [-1, 128, 16, 16]               0
      BatchNorm2d-17          [-1, 128, 16, 16]             256
             ReLU-18          [-1, 128, 16, 16]               0
             ReLU-18          [-1, 128, 16, 16]               0
AdaptiveAvgPool2d-19            [-1, 128, 1, 1]               0
AdaptiveAvgPool2d-19            [-1, 128, 1, 1]               0
           Linear-20                   [-1, 10]           1,290
           Linear-20                   [-1, 10]           1,290
================================================================
Total params: 22,554
Trainable params: 22,554
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.81
Params size (MB): 0.09
Estimated Total Size (MB): 6.91
