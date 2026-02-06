# dataset.py
# Loads images and their .pts landmark files for training.

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

def read_pts_file(filepath):
    """
    Reads a .pts landmark file and returns a numpy array of shape (55, 2)
    """
    points = []
    with open(filepath, "r") as f:
        lines = f.readlines()

        # Landmark points start after the '{' line
        start = lines.index('{\n') + 1
        end = lines.index('}\n')

        for line in lines[start:end]:
            x, y = map(float, line.strip().split())
            points.append([x, y])

    return np.array(points, dtype=np.float32)

class EarDataset(Dataset):
    def __init__(self, folder, augment=False):
        """
        folder: path to dataset/train or dataset/test
        augment: whether to apply data augmentation (for training)
        """
        self.folder = folder
        self.images = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image filename
        img_name = self.images[idx]
        img_path = os.path.join(self.folder, img_name)

        # Corresponding landmark file
        pts_path = img_path.replace(".png", ".pts")

        # Load image
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # Load landmarks
        landmarks = read_pts_file(pts_path)

        # Apply augmentation if training
        if self.augment:
            # Random brightness/contrast adjustment
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  # contrast
                beta = random.randint(-20, 20)     # brightness
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # Random horizontal flip (for left/right ear pairs)
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                landmarks[:, 0] = w - landmarks[:, 0]

        # Resize image to 224x224 for ResNet
        img_resized = cv2.resize(img, (224, 224))
        img_resized = img_resized / 255.0  # normalize pixels
        img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC → CHW

        # Normalize landmarks to [0,1]
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h

        # Flatten landmarks into 110-length vector
        landmarks = landmarks.flatten()

        return torch.tensor(img_resized, dtype=torch.float32), torch.tensor(landmarks, dtype=torch.float32)
