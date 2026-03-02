import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

from infant_dataset import get_train_test_split
from model import get_model, soft_argmax_2d

NUM_LANDMARKS = 22
NUM_STAGES = 6

CKPT_PATH = f"infant_model_{NUM_LANDMARKS}lm_best.pth"
OUTPUT_DIR = f"infant_eval_results_{NUM_LANDMARKS}"

IMAGES_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/images"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(NUM_LANDMARKS, NUM_STAGES).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()
print(f"Loaded checkpoint: {CKPT_PATH}")

_, test_dataset = get_train_test_split(num_landmarks=NUM_LANDMARKS)
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


all_errors = np.array(all_errors)
per_landmark_errors = np.array(per_landmark_errors)

avg_per_landmark = per_landmark_errors.mean(axis=0)
worst_idx = avg_per_landmark.argmax()
best_idx = avg_per_landmark.argmin()

print(f"------- Error statistics for normalized coordinates (0-1) -------")
print(f"Average error:  {all_errors.mean():.6f}")
print(f"Median error:   {np.median(all_errors):.6f}")
print(f"Std deviation:  {all_errors.std():.6f}")
print(f"Min error:      {all_errors.min():.6f}")
print(f"Max error:      {all_errors.max():.6f}")

print(f"------- Best and worst landmarks -------")
print(f"Best landmark  (#{best_idx+1}):  {avg_per_landmark[best_idx]:.6f}")
print(f"Worst landmark (#{worst_idx+1}): {avg_per_landmark[worst_idx]:.6f}")