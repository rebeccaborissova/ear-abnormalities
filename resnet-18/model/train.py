# train.py
# Trains multi-stage heatmap model for ear landmarks .

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7" 
# run 'nvidia-smi' in terminal to check GPU usage

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import EarDataset
from model import get_model
from utils import save_model

# Number of landmarks
NUM_LANDMARKS = 55
NUM_STAGES = 6

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("Using cuda")
else:
    print("Using cpu")

# Load dataset with heatmap targets
train_dataset = EarDataset(
    "/home/UFAD/mansapatel/CollectionA/train",
    augment=True,
    input_size=368,
    heatmap_size=46
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

# Create model
model = get_model(NUM_LANDMARKS, NUM_STAGES).to(device)

# Multi-stage MSE loss 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

best_loss = float('inf')

print(f"Training with {len(train_dataset)} images")
print(f"Input size: 368x368, Heatmap size: 46x46")
print(f"Number of stages: {NUM_STAGES}\n")

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch_idx, (imgs, target_heatmaps) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1}")
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
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")
    print(f"  Current learning rate: {optimizer.param_groups[0]['lr']:.6e}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_model(model, "ear_landmark_model_best.pth")
        print(f"  → Saved best model (loss: {best_loss:.6f})")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_model(model, f"ear_landmark_model_epoch_{epoch+1}.pth")
        print(f"  → Saved checkpoint at epoch {epoch+1}")
    
    # Step scheduler
    scheduler.step()



