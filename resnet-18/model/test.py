import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # pick a free GPU

import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import get_model
from utils import load_model
from dataset import EarDataset

NUM_LANDMARKS = 55  # Must match training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = get_model(NUM_LANDMARKS).to(device)
model = load_model(model, "ear_landmark_model_best_new.pth")
model.eval()

print("Using new improved model architecture")
print("Loading best model checkpoint: ear_landmark_model_best_new.pth\n")

# Load test dataset
test_dataset = EarDataset("../dataset/test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Testing on {len(test_dataset)} images...")

total_error = 0
errors = []
per_landmark_errors = []

with torch.no_grad():
    for idx, (imgs, landmarks) in enumerate(tqdm(test_loader)):
        imgs = imgs.to(device)
        landmarks = landmarks.cpu().numpy().reshape(-1, 2)
        
        # Predict
        preds = model(imgs).cpu().numpy().reshape(-1, 2)
        
        # Calculate error per landmark (Euclidean distance)
        landmark_errors = np.sqrt(np.sum((preds - landmarks) ** 2, axis=1))
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

# Visualize predictions on first 5 images
print("\nVisualizing predictions on first 5 images...")
output_dir = "../test_results"
os.makedirs(output_dir, exist_ok=True)

for idx in range(min(10, len(test_dataset))):
    # Get image and landmarks from dataset
    img_tensor, landmarks_gt = test_dataset[idx]
    
    # Construct image path
    img_path = os.path.join(test_dataset.folder, test_dataset.images[idx])
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # Prepare input
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        pred = model(img_tensor).cpu().detach().numpy().reshape(-1, 2)
    
    # Denormalize predictions
    pred[:, 0] *= w
    pred[:, 1] *= h
    
    # Denormalize ground truth
    landmarks_gt = landmarks_gt.numpy().reshape(-1, 2)
    landmarks_gt[:, 0] *= w
    landmarks_gt[:, 1] *= h
    
    # Draw predictions (red) and ground truth (green)
    for (x, y) in pred:
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    for (x, y) in landmarks_gt:
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    # Save image
    output_path = os.path.join(output_dir, f"result_{idx+1}.png")
    cv2.imwrite(output_path, img)
    print(f"Saved visualization to {output_path}")

print(f"\nAll visualizations saved to {output_dir}/")
