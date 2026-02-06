# train.py
# Trains ResNet-18 to predict ear landmarks.

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # pick a free GPU
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

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("Using cuda")
else:
    print("Using cpu")

# Load dataset
train_dataset = EarDataset("../dataset/train", augment=True)  # Enable augmentation
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

# Create model
model = get_model(NUM_LANDMARKS).to(device)

# Use SmoothL1Loss for better robustness
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

# Add learning rate scheduler with cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

best_loss = float('inf')

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch_idx, (imgs, landmarks) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1}")
    ):
        imgs = imgs.to(device)
        landmarks = landmarks.to(device)

        preds = model(imgs)
        loss = criterion(preds, landmarks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        scheduler.step(epoch + batch_idx / len(train_loader))

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


