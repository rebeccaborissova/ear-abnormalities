import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  

import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import get_model, soft_argmax_2d
from utils import load_model
from dataset import EarDataset

NUM_LANDMARKS = 55  
NUM_STAGES = 6
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = get_model(NUM_LANDMARKS, NUM_STAGES).to(device)
model = load_model(model, "ear_landmark_model_best.pth")
model.eval()

print("Using multi-stage heatmap model (OpenPose-style)")
print("Loading best model checkpoint: ear_landmark_model_best.pth\n")

# Load test dataset
test_dataset = EarDataset(
    "/home/UFAD/rborissova/senior_project/data_test_combined",
    augment=False,
    input_size=368,
    heatmap_size=46
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Testing on {len(test_dataset)} images...")

total_error = 0
errors = []
per_landmark_errors = []

with torch.no_grad():
    for idx, (imgs, target_heatmaps) in enumerate(tqdm(test_loader)):
        imgs = imgs.to(device)
        
        # Get ground truth coordinates from heatmaps
        landmarks_gt = soft_argmax_2d(target_heatmaps, normalize=True).cpu().numpy().squeeze()
        
        # Predict: get all stages, use last stage
        stage_outputs = model(imgs)
        final_heatmaps = stage_outputs[:, -1, :, :, :]  # Last stage
        
        # Convert heatmaps to coordinates
        preds = soft_argmax_2d(final_heatmaps, normalize=True).cpu().numpy().squeeze()
        
        # Calculate error per landmark (Euclidean distance in normalized space)
        landmark_errors = np.sqrt(np.sum((preds - landmarks_gt) ** 2, axis=1))
        per_landmark_errors.append(landmark_errors)
        
        # Mean error for this image
        error = landmark_errors.mean()
        errors.append(error)
        total_error += error

# Convert to numpy array for analysis
per_landmark_errors = np.array(per_landmark_errors)  # Shape: (num_images, num_landmarks)
errors = np.array(errors)

# Calculate comprehensive metrics
avg_error = errors.mean()
median_error = np.median(errors)
min_error = errors.min()
max_error = errors.max()
std_error = errors.std()

# Per-landmark statistics
avg_per_landmark = per_landmark_errors.mean(axis=0)  # Average error per landmark across all images
worst_landmark_idx = avg_per_landmark.argmax()
best_landmark_idx = avg_per_landmark.argmin()

# Calculate percentage accuracy at different thresholds
thresholds = [0.01, 0.015, 0.02, 0.03, 0.05, 0.10]  # normalized coordinate thresholds
print(f"\n{'='*60}")
print(f"PERCENTAGE ACCURACY (Landmarks within threshold)")
print(f"{'='*60}")
for threshold in thresholds:
    # Count landmarks within threshold
    within_threshold = (per_landmark_errors < threshold).sum()
    total_landmarks = per_landmark_errors.size
    accuracy_pct = (within_threshold / total_landmarks) * 100
    print(f"Within {threshold:.3f} units: {accuracy_pct:.2f}% ({within_threshold}/{total_landmarks} landmarks)")

# Overall accuracy - using 0.015 as default threshold
default_threshold = 0.015
within_default = (per_landmark_errors < default_threshold).sum()
overall_accuracy = (within_default / per_landmark_errors.size) * 100
print(f"\n{'='*60}")
print(f"OVERALL ACCURACY: {overall_accuracy:.2f}%")
print(f"(at threshold = {default_threshold} normalized units)")
print(f"{'='*60}")

# Print detailed metrics
print(f"\n{'='*60}")
print(f"ERROR STATISTICS")
print(f"{'='*60}")
print(f"Number of test images: {len(test_dataset)}")
print(f"Number of landmarks per image: {NUM_LANDMARKS}")
print(f"\nError Statistics (normalized coordinates):")
print(f"  Average error: {avg_error:.6f}")
print(f"  Median error:  {median_error:.6f}")
print(f"  Std deviation: {std_error:.6f}")
print(f"  Min error:     {min_error:.6f}")
print(f"  Max error:     {max_error:.6f}")

# Percentile analysis
percentiles = [25, 50, 75, 90, 95, 99]
print(f"\nError Percentiles:")
for p in percentiles:
    print(f"  {p}th percentile: {np.percentile(errors, p):.6f}")

# Per-landmark analysis
print(f"\n{'='*60}")
print(f"PER-LANDMARK ANALYSIS")
print(f"{'='*60}")
print(f"Best performing landmark (#{best_landmark_idx+1}): {avg_per_landmark[best_landmark_idx]:.6f}")
print(f"Worst performing landmark (#{worst_landmark_idx+1}): {avg_per_landmark[worst_landmark_idx]:.6f}")
print(f"\nTop 5 most accurate landmarks:")
best_landmarks = avg_per_landmark.argsort()[:5]
for i, lm_idx in enumerate(best_landmarks, 1):
    print(f"  {i}. Landmark #{lm_idx+1}: {avg_per_landmark[lm_idx]:.6f}")
print(f"\nTop 5 least accurate landmarks:")
worst_landmarks = avg_per_landmark.argsort()[-5:][::-1]
for i, lm_idx in enumerate(worst_landmarks, 1):
    print(f"  {i}. Landmark #{lm_idx+1}: {avg_per_landmark[lm_idx]:.6f}")

# Visualize predictions on images 11-20
print("\nVisualizing predictions on images 11-20...")
output_dir = "../test_results"
os.makedirs(output_dir, exist_ok=True)

for idx in range(10, min(20, len(test_dataset))):
    # Load original image (before any processing)
    img_name = test_dataset.images[idx]
    img_path = os.path.join(test_dataset.folder, img_name)
    pts_path = img_path.replace(".png", ".txt")
    
    # Read original image and landmarks
    from dataset import read_pts_file
    img_orig = cv2.imread(img_path)
    landmarks_orig = read_pts_file(pts_path)
    
    # Get cropped/processed image for prediction
    img_tensor, target_heatmap = test_dataset[idx]
    
    # Get ground truth from heatmap
    landmarks_gt = soft_argmax_2d(target_heatmap.unsqueeze(0), normalize=True).cpu().numpy().squeeze()
    
    # Prepare input
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        stage_outputs = model(img_tensor)
        final_heatmaps = stage_outputs[:, -1, :, :, :]
        pred = soft_argmax_2d(final_heatmaps, normalize=True).cpu().numpy().squeeze()
    
    # For visualization: work on the cropped image since that's what the model saw
    # Recreate the cropped image to match what dataset returns
    h_orig, w_orig, _ = img_orig.shape
    
    # Apply same cropping as dataset
    x_min, y_min = landmarks_orig.min(axis=0)
    x_max, y_max = landmarks_orig.max(axis=0)
    ear_w = x_max - x_min
    ear_h = y_max - y_min
    pad = 10
    s_x = max(int(x_min - ear_w * pad), 0)
    e_x = min(int(x_max + ear_w * pad), w_orig)
    s_y = max(int(y_min - ear_h * pad), 0)
    e_y = min(int(y_max + ear_h * pad), h_orig)
    
    img_crop = img_orig[s_y:e_y, s_x:e_x, :]
    h_crop, w_crop, _ = img_crop.shape
    
    # Denormalize predictions and ground truth to cropped image space
    pred_px = pred * np.array([w_crop - 1, h_crop - 1])
    gt_px = landmarks_gt * np.array([w_crop - 1, h_crop - 1])
    
    # Draw predictions (red) and ground truth (green)
    for (x, y) in pred_px:
        cv2.circle(img_crop, (int(x), int(y)), 2, (0, 0, 255), -1)
    for (x, y) in gt_px:
        cv2.circle(img_crop, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    # Save image
    output_path = os.path.join(output_dir, f"result_{idx+1}.png")
    cv2.imwrite(output_path, img_crop)
    print(f"Saved visualization to {output_path}")

print(f"\nAll visualizations saved to {output_dir}/")
