import os
import torch
import cv2
import numpy as np

from model import get_model, soft_argmax_2d
from utils import load_model

NUM_LANDMARKS = 23

# Directories/files
MODEL_PATH = f"infant_ear_model_{NUM_LANDMARKS}lm_best.pth"
IMAGES_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/images"
OUTPUT_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/landmarks/predicted"

NUM_STAGES = 6
INPUT_SIZE = 368

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(NUM_LANDMARKS, NUM_STAGES).to(device)
model = load_model(model, MODEL_PATH)
model.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)

img_paths = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
print(f"Found {len(img_paths)} images")

for img_path in img_paths:
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]  # e.g. "0031_L"
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not read {img_path}!")
        continue

    h, w = img.shape[:2]

    # Preprocess the same way as dataset.py
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_resized = img_resized / 255.0
    img_chw = np.transpose(img_resized, (2, 0, 1))
    img_tensor = torch.tensor(img_chw, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        stage_outputs = model(img_tensor)
        final_heatmaps = stage_outputs[:, -1]
        pred = soft_argmax_2d(final_heatmaps, normalize=True).cpu().numpy().squeeze()

    # Denormalize to original image size
    pred_px = pred * np.array([w - 1, h - 1])

    # Write .txt landmark file
    txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
    with open(txt_path, "w") as f:
        f.write("version: 1\n")
        f.write(f"n_points: {NUM_LANDMARKS}\n")
        f.write("{\n")
        for (x, y) in pred_px:
            f.write(f"{int(round(x))} {int(round(y))}\n")
        f.write("}\n")

    print(f"Saved: {txt_path}")