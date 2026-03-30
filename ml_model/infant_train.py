# infant_train.py
# Trains infant ear landmark model with transfer learning

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import adult_model
from infant_dataset import get_train_test_split


def train_infant_model(config):
    """
    Train infant ear landmark model
    
    Args:
        config: Dictionary with training configuration
    """
    # Extract config
    NUM_LANDMARKS = config['num_landmarks']
    NUM_STAGES = config['num_stages']
    BATCH_SIZE = config['batch_size']
    NUM_EPOCHS = config['num_epochs']
    LEARNING_RATE = float(config['learning_rate'])
    WEIGHT_DECAY = float(config['weight_decay'])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model with pretrained weights
    net = adult_model.get_model(NUM_LANDMARKS, NUM_STAGES).to(device)
    
    if config['input_checkpoint']:
        print(f"Loading pretrained weights from {config['input_checkpoint']}")
        net.load_state_dict(torch.load(config['input_checkpoint'], map_location=device))
    
    #TESTING THIS
    # FREEZE BACKBONE - only train heatmap stages
    for param in net.feature_extractor.parameters():
        param.requires_grad = False
    print("Feature extractor frozen. Training only heatmap stages.")

    # Load datasets
    train_dataset, test_dataset = get_train_test_split(num_landmarks=NUM_LANDMARKS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler_config = config['scheduler']
    if scheduler_config['type'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=scheduler_config['patience'],
            factor=scheduler_config['factor']
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_config['type']}")
    
    best_val_loss = float("inf")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        net.train()
        train_loss = 0.0
        
        for imgs, target_heatmaps in train_loader:
            imgs = imgs.to(device)
            target_heatmaps = target_heatmaps.to(device)
            
            optimizer.zero_grad()
            stage_outputs = net(imgs)
            
            # Multi-stage loss
            loss = 0.0
            for s in range(NUM_STAGES):
                loss += criterion(stage_outputs[:, s], target_heatmaps)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        net.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for imgs, target_heatmaps in test_loader:
                imgs = imgs.to(device)
                target_heatmaps = target_heatmaps.to(device)
                
                stage_outputs = net(imgs)
                # Use only final stage for validation
                loss = criterion(stage_outputs[:, -1], target_heatmaps)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), config['output_checkpoint'])
            print(f"  → Saved new best checkpoint! Val loss: {val_loss:.6f}")
    
    print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
    return best_val_loss


# Standalone execution (backwards compatible)
if __name__ == "__main__":
    # Original standalone config
    config = {
        'num_landmarks': 23,
        'num_stages': 6,
        'input_checkpoint': 'infant_ear_model_23lm.pth',
        'output_checkpoint': 'infant_ear_model_23lm_best_v4.pth',
        'batch_size': 4,
        'num_epochs': 150,
        'learning_rate': 5e-5,
        'weight_decay': 5e-4,
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'patience': 20,
            'factor': 0.3
        }
    }
    
    # Set environment variables
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    
    train_infant_model(config)
