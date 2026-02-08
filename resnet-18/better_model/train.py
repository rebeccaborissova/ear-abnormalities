# train.py

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["OMP_NUM_THREADS"] = "1"


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from dataset import EarDataset
from model import EarLandmarkModel
from config import *

# --------------------------------------------------
# Collate function (pads images only)
# --------------------------------------------------
def collate_fn(batch):
    imgs, landmarks, sizes = zip(*batch)

    max_h = max(s[0] for s in sizes)
    max_w = max(s[1] for s in sizes)

    padded_imgs = []
    for img in imgs:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded_imgs.append(F.pad(img, (0, pad_w, 0, pad_h)))

    return (
        torch.stack(padded_imgs),
        torch.stack(landmarks)
    )

# --------------------------------------------------
# Data
# --------------------------------------------------
train_dataset = EarDataset(TRAIN_IMG_DIR)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=collate_fn
)

# --------------------------------------------------
# Model / optimization
# --------------------------------------------------
model = EarLandmarkModel().to(DEVICE)

criterion = nn.SmoothL1Loss(beta=1.0)
optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scaler = GradScaler(enabled=USE_AMP)

# --------------------------------------------------
# Training
# --------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, landmarks in train_loader:
        imgs = imgs.to(DEVICE)
        landmarks = landmarks.to(DEVICE)

        optimizer.zero_grad()

        with autocast(enabled=USE_AMP):
            preds = model(imgs)
            loss = criterion(preds, landmarks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss/len(train_loader):.6f}"
    )

torch.save(
    model.state_dict(),
    f"{CHECKPOINT_DIR}/ear_landmarks_effnet.pth"
)
