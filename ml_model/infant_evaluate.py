# This file is NOT part of the pipeline. Use it to test the model

import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

from infant_dataset import get_train_test_split
from adult_model import get_model, soft_argmax_2d

# ***************************************************************************************************** 

# UPDATE THESE VARIABLES BEFORE RUNNING:
INPUT_MODEL_PATH = f"infant_ear_model_23lm_best_v2.pth" # name of the trained infant ear model
OUTPUT_DIR = f"infant_eval_results_23" # directory name to save test images in
IMAGES_DIR = "/home/UFAD/angelali/ears/images/images" # location of infant ear images
SPECIFIC_IMAGES = ["0148_R.jpg", "0080_R.jpg"] # specific images to test (in addition to test dataset)

# ***************************************************************************************************** 

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(23, 6).to(device)
model.load_state_dict(torch.load(INPUT_MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded checkpoint: {INPUT_MODEL_PATH}")

_, test_dataset = get_train_test_split(num_landmarks=23)
test_loader = DataLoader(test_dataset)

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_errors = []
per_landmark_errors = []

with torch.no_grad():
    for idx, (imgs, target_heatmaps) in enumerate(test_loader):
        imgs = imgs.to(device)

        # Ground truth coordinates
        landmarks_gt = soft_argmax_2d(target_heatmaps, normalize=True).cpu().numpy().squeeze()

        # Predicted coordinates
        pred = soft_argmax_2d(model(imgs)[:, -1], normalize=True).cpu().numpy().squeeze()

        # Compute euclidean distance error for each landmark
        landmark_errors = np.sqrt(np.sum((pred - landmarks_gt) ** 2, axis=1))
        per_landmark_errors.append(landmark_errors)
        all_errors.append(landmark_errors.mean())

        # Plot predicted vs. ground truth points on original image (red = pred, green = ground truth)
        img_name = test_dataset.image_files[idx]
        img = cv2.imread(os.path.join(IMAGES_DIR, img_name))
        h, w = img.shape[:2]

        pred_px = pred * np.array([w - 1, h - 1])
        gt_px   = landmarks_gt * np.array([w - 1, h - 1])

        for (x, y) in pred_px:
            cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)
        for (x, y) in gt_px:
            cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)

        out_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
        cv2.imwrite(out_path, img)

# Test specific images directly from IMAGES_DIR
if SPECIFIC_IMAGES:
    print(f"\nTesting specific images from {IMAGES_DIR}:")
    for img_name in SPECIFIC_IMAGES:
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"  {img_name}: NOT FOUND")
            continue
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # Normalize and prepare image for model
        img_resized = cv2.resize(img, (368, 368))
        img_normalized = img_resized / 255.0
        img_tensor = np.transpose(img_normalized, (2, 0, 1))
        img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            pred = soft_argmax_2d(model(img_tensor)[:, -1], normalize=True).cpu().numpy().squeeze()
        
        # Draw predictions on original image
        img_output = img.copy()
        pred_px = pred * np.array([w - 1, h - 1])
        for (x, y) in pred_px:
            cv2.circle(img_output, (int(x), int(y)), 6, (0, 0, 255), -1)
        
        out_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
        cv2.imwrite(out_path, img_output)
        print(f"  {img_name}: Tested and saved to {out_path}")

all_errors = np.array(all_errors)
per_landmark_errors = np.array(per_landmark_errors)

if len(all_errors) > 0:
    avg_per_landmark = per_landmark_errors.mean(axis=0)
    worst_idx = avg_per_landmark.argmax()
    best_idx = avg_per_landmark.argmin()

    print(f"\n------- Error statistics for normalized coordinates (0-1) -------")
    print(f"Average error:  {all_errors.mean():.6f}")
    print(f"Median error:   {np.median(all_errors):.6f}")
    print(f"Std deviation:  {all_errors.std():.6f}")
    print(f"Min error:      {all_errors.min():.6f}")
    print(f"Max error:      {all_errors.max():.6f}")

else:
    print("No test set images were evaluated.")