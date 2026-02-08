# test.py

import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F

from model import EarLandmarkModel
from utils import load_pts
from config import *

model = EarLandmarkModel().to(DEVICE)
model.load_state_dict(
    torch.load(f"{CHECKPOINT_DIR}/ear_landmarks_effnet.pth")
)
model.eval()

test_files = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".png")]
)

all_nme, all_pck = [], []

# Create output directory for visualizations
vis_output_dir = "/home/UFAD/angelali/ears/test_visualizations"
os.makedirs(vis_output_dir, exist_ok=True)

for idx, img_name in enumerate(test_files):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    pts_path = img_path.replace(".png", ".pts")

    img = Image.open(img_path).convert("RGB")
    gt = load_pts(pts_path)

    orig_w, orig_h = img.size

    # --------------------------------------------------
    # Same scaling as training
    # --------------------------------------------------
    scale = IMG_SIZE / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    img_tensor = F.normalize(
        F.to_tensor(img_resized),
        mean=[0.5]*3,
        std=[0.5]*3
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor).cpu().view(NUM_LANDMARKS, 2).numpy()

    # --------------------------------------------------
    # Back to original image space
    # --------------------------------------------------
    pred /= scale

    norm = np.sqrt(orig_w**2 + orig_h**2)
    dist = np.linalg.norm(pred - gt, axis=1)

    all_nme.append(np.mean(dist) / norm)
    all_pck.append(np.mean(dist < 0.05 * norm) * 100)

    # --------------------------------------------------
    # Visualize first 10 samples
    # --------------------------------------------------
    if idx < 10:
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        
        # Draw ground truth in green
        for x, y in gt:
            draw.ellipse([x-3, y-3, x+3, y+3], fill='green', outline='green')
        
        # Draw predictions in red
        for x, y in pred:
            draw.ellipse([x-3, y-3, x+3, y+3], fill='red', outline='red')
        
        output_path = os.path.join(vis_output_dir, f"vis_{img_name}")
        vis_img.save(output_path)

print(f"\nSaved sample visualizations to {vis_output_dir}")
print(f"\nMean NME: {np.mean(all_nme):.4f}")
print(f"Mean PCK@0.05: {np.mean(all_pck):.2f}%")
