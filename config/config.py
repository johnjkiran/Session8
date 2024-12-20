class Config:
    # Dataset
    NUM_CLASSES = 10
    IMG_SIZE = 32
    BATCH_SIZE = 128
    
    # Training
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = 'cuda'
    
    # Augmentation
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2616] 