import torch
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class AlbumentationsTransform:
    def __init__(self, mean, std):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(
                max_holes=1, max_height=16, max_width=16,
                min_holes=1, min_height=16, min_width=16,
                fill_value=mean, p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.transform(image=img)['image']

def get_dataloaders(config):
    # Training transforms
    train_transform = AlbumentationsTransform(config.MEAN, config.STD)
    
    # Test transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])
    
    # Datasets
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=test_transform
    )
    
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, 
        shuffle=True, num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader 