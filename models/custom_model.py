import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                  padding=dilation, groups=in_channels, 
                                  stride=stride, dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # C1: Regular Conv (reduced channels)
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        
        # C2: Depthwise Separable Conv
        self.c2 = nn.Sequential(
            DepthwiseSeparableConv(24, 48),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # C3: Dilated Depthwise Separable Conv
        self.c3 = nn.Sequential(
            DepthwiseSeparableConv(48, 96, dilation=2),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        
        # C4: Strided Depthwise Separable Conv
        self.c4 = nn.Sequential(
            DepthwiseSeparableConv(96, 128, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final FC Layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x 