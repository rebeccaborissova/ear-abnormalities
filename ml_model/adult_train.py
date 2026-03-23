import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from adult_dataset import EarDataset
from adult_model import get_model
from adult_utils import save_model


def train_adult_model(config):
    """
    Train adult ear landmark model
    
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
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset with heatmap targets
    train_dataset = EarDataset(
        config['dataset_path'],
        augment=True,
        input_size=config['input_size'],
        heatmap_size=config['heatmap_size']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = get_model(NUM_LANDMARKS, NUM_STAGES).to(device)
    
    # Multi-stage MSE loss 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler_config = config['scheduler']
    if scheduler_config['type'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_config['type']}")
    
    best_loss = float('inf')
    
    print(f"Training with {len(train_dataset)} images")
    print(f"Input size: {config['input_size']}x{config['input_size']}, "
          f"Heatmap size: {config['heatmap_size']}x{config['heatmap_size']}")
    print(f"Number of stages: {NUM_STAGES}\n")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (imgs, target_heatmaps) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        ):
            imgs = imgs.to(device)
            target_heatmaps = target_heatmaps.to(device)
            
            # Forward pass: get all stage outputs
            # Shape: (batch, num_stages, num_landmarks, H, W)
            stage_outputs = model(imgs)
            
            # Multi-stage loss: sum loss from all stages
            loss = 0
            for stage_idx in range(NUM_STAGES):
                stage_heatmaps = stage_outputs[:, stage_idx, :, :, :]
                # Resize to match target size if needed
                if stage_heatmaps.shape[-2:] != target_heatmaps.shape[-2:]:
                    stage_heatmaps = torch.nn.functional.interpolate(
                        stage_heatmaps,
                        size=target_heatmaps.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                loss += criterion(stage_heatmaps, target_heatmaps)
            
            # Average loss across stages
            loss = loss / NUM_STAGES
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.6f}")
        print(f"  Current learning rate: {optimizer.param_groups[0]['lr']:.6e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, config['output_checkpoint'])
            print(f"  → Saved best model (loss: {best_loss:.6f})")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_name = config['output_checkpoint'].replace('.pth', f'_epoch_{epoch+1}.pth')
            save_model(model, checkpoint_name)
            print(f"  → Saved checkpoint at epoch {epoch+1}")
        
        # Step scheduler
        scheduler.step()
    
    print(f"\nTraining completed. Best loss: {best_loss:.6f}")
    return best_loss


# Standalone execution (backwards compatible)
if __name__ == "__main__":
    # Original standalone config
    config = {
        'num_landmarks': 55,
        'num_stages': 6,
        'dataset_path': "/home/UFAD/angelali/ears/dataset/train",
        'input_size': 368,
        'heatmap_size': 46,
        'batch_size': 8,
        'num_workers': 2,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'scheduler': {
            'type': 'StepLR',
            'step_size': 30,
            'gamma': 0.5
        },
        'output_checkpoint': 'ear_landmark_model_best.pth',
        'save_interval': 10
    }
    
    # Set environment variables
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    
    train_adult_model(config)
