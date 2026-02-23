# dataset.py
# Loads images and their .txt landmark files for training with heatmap targets.

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

def read_pts_file(filepath):
    """
    Reads a .txt landmark file and returns a numpy array of shape (55, 2)
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


def make_gaussian_heatmap(landmarks, height, width, sigma=2.5):
    """
    Generate Gaussian heatmaps for landmarks (OpenPose-style).
    
    Args:
        landmarks: (K, 2) array of normalized [0,1] coordinates
        height: heatmap height
        width: heatmap width
        sigma: Gaussian sigma
    
    Returns:
        heatmaps: (K, H, W) array of confidence maps
    """
    num_landmarks = landmarks.shape[0]
    heatmaps = np.zeros((num_landmarks, height, width), dtype=np.float32)
    
    # Convert normalized coords to heatmap coords
    landmarks_px = landmarks * np.array([width - 1, height - 1])
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    
    for i, (x, y) in enumerate(landmarks_px):
        # Gaussian centered at (x, y)
        dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
        heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
        heatmaps[i] = heatmap
    
    return heatmaps

class EarDataset(Dataset):
    def __init__(self, folder, augment=False, input_size=368, heatmap_size=46):
        """
        folder: path to dataset/train or dataset/test
        augment: whether to apply data augmentation (for training)
        input_size: input image size (default: 368 like OpenPose)
        heatmap_size: target heatmap size (default: 46 for 1/8 resolution)
        """
        self.folder = folder
        self.augment = augment
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        
        image_files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
        self.images = []
        for filename in image_files:
            img_path = os.path.join(folder, filename)
            pts_path = img_path.replace(".png", ".txt")
            if not os.path.isfile(pts_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            self.images.append(filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image filename
        img_name = self.images[idx]
        img_path = os.path.join(self.folder, img_name)

        # Corresponding landmark file
        pts_path = img_path.replace(".png", ".txt")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h, w, _ = img.shape

        # Load landmarks
        landmarks = read_pts_file(pts_path)

        # Apply ear-aware cropping with padding (like TensorFlow version)
        img, landmarks = self._crop_to_ear(img, landmarks)
        h, w, _ = img.shape
        
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
            
            # Random shrinking (like TensorFlow version)
            if random.random() > 0.5:
                img, landmarks = self._shrink_image(img, landmarks)
                h, w, _ = img.shape

        # Normalize landmarks to [0,1]
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h
        
        # Clip to valid range
        landmarks = np.clip(landmarks, 0, 1)
        
        # Resize image to input size
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_resized = img_resized / 255.0  # normalize pixels
        img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC → CHW
        
        # Generate Gaussian heatmaps
        heatmaps = make_gaussian_heatmap(landmarks, self.heatmap_size, self.heatmap_size, sigma=2.5)

        return (
            torch.tensor(img_resized, dtype=torch.float32),
            torch.tensor(heatmaps, dtype=torch.float32)
        )
    
    def _crop_to_ear(self, img, landmarks):
        """Crop image to ear region with padding (like TensorFlow version)"""
        h, w, _ = img.shape
        
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        
        ear_w = x_max - x_min
        ear_h = y_max - y_min
        
        # Add generous padding (10x like TF version for full context)
        pad = 10
        s_x = max(int(x_min - ear_w * pad), 0)
        e_x = min(int(x_max + ear_w * pad), w)
        s_y = max(int(y_min - ear_h * pad), 0)
        e_y = min(int(y_max + ear_h * pad), h)
        
        # Crop
        img_crop = img[s_y:e_y, s_x:e_x, :]
        landmarks_crop = landmarks - np.array([s_x, s_y])
        
        return img_crop, landmarks_crop
    
    def _shrink_image(self, img, landmarks):
        """Random horizontal shrinking (like TensorFlow version)"""
        h, w, _ = img.shape
        
        x_min = int(np.min(landmarks[:, 0]))
        x_max = int(np.max(landmarks[:, 0]))
        
        # Random shrink ratio
        max_ratio = 4
        sh_ratio = np.random.randint(1, max_ratio)
        
        # Split and shrink middle section
        image_left = img[:, :x_min, :]
        image_right = img[:, x_max:, :]
        image_mid = img[:, x_min:x_max:sh_ratio, :]
        
        sh_img = np.concatenate((image_left, image_mid, image_right), axis=1)
        sh_landmarks = (landmarks - np.array([x_min, 0])) / np.array([sh_ratio, 1]) + np.array([x_min, 0])
        
        return sh_img, sh_landmarks

