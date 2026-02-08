# dataset.py
# Aspect-ratio-safe landmark dataset (NO distortion)

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

from utils import load_pts
from config import *

class EarDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".png")]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        pts_path = img_path.replace(".png", ".pts")

        # Load raw
        img = Image.open(img_path).convert("RGB")
        landmarks = load_pts(pts_path)

        orig_w, orig_h = img.size

        # --------------------------------------------------
        # Uniform scaling (longest side → IMG_SIZE)
        # --------------------------------------------------
        scale = IMG_SIZE / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        img = img.resize((new_w, new_h), Image.BILINEAR)

        landmarks = landmarks.copy()
        landmarks[:, 0] *= scale
        landmarks[:, 1] *= scale

        # --------------------------------------------------
        # To tensor
        # --------------------------------------------------
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.5]*3, std=[0.5]*3)

        landmarks = torch.tensor(landmarks, dtype=torch.float32).view(-1)

        return img, landmarks, (new_h, new_w)
