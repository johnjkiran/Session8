import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from config.config import Config
from models.custom_model import CIFAR10Net
from utils.data_loader import get_dataloaders
from utils.training import train_epoch, test_epoch

def main():
    # Setup
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = CIFAR10Net(Config.NUM_CLASSES).to(device)
    
    # Print model summary
    print("\nModel Summary:")
    summary(model, input_size=(3, 32, 32))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(Config)
    
    # Training loop
    best_acc = 0
    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch: {epoch+1}/{Config.EPOCHS}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Test
        test_loss, test_acc = test_epoch(
            model, test_loader, criterion, device
        )
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    main() 