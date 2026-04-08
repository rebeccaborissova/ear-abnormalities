import os
import sys
import csv
import yaml
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adult_model import get_model, soft_argmax_2d
from infant_dataset import get_train_test_split

with open("../config.yaml") as f:
    config = yaml.safe_load(f)

BASE_DIR        = "/home/UFAD/jingyifu/ear-project/ear-abnormalities/resnet-18/model"
CKPT_23         = os.path.join(BASE_DIR, "infant_ear_model_23lm_best.pth")
OUTPUT_DIR      = "test_outputs/accuracy"
VIS_DIR         = "test_outputs/accuracy/visualizations"
IMAGES_DIR      = os.environ.get("INFANT_IMAGES_DIR", "")
ERROR_THRESHOLD = 0.25
NUM_LANDMARKS   = 23
NUM_STAGES      = 6
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

PASSED = 0
FAILED = 0

def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        print(f"  [PASS] {name}")
        PASSED += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        FAILED += 1

print("\n======== Accuracy Evaluation ========")
model = get_model(NUM_LANDMARKS, NUM_STAGES).to(DEVICE)
if os.path.exists(CKPT_23):
    model.load_state_dict(torch.load(CKPT_23, map_location=DEVICE))
    print(f"  Loaded {CKPT_23}")
else:
    print(f"  WARNING: {CKPT_23} not found")
model.eval()

_, test_dataset = get_train_test_split(config, num_landmarks=NUM_LANDMARKS)
test_loader     = DataLoader(test_dataset, batch_size=1, shuffle=False)

all_errors = []
rows       = []

with torch.no_grad():
    for idx, (imgs, target_heatmaps) in enumerate(test_loader):
        imgs     = imgs.to(DEVICE)
        gt       = soft_argmax_2d(target_heatmaps, normalize=True).cpu().numpy().squeeze()
        out      = model(imgs)
        heat     = out[:, -1] if out.ndim == 5 else out
        pred     = soft_argmax_2d(heat, normalize=True).cpu().numpy().squeeze()
        errors   = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
        all_errors.append(errors.mean())

        img_name = test_dataset.image_files[idx]
        orig = cv2.imread(os.path.join(IMAGES_DIR, img_name))
        if orig is not None:
            H, W = orig.shape[:2]
            for x, y in pred * np.array([W - 1, H - 1]):
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
            for x, y in gt * np.array([W - 1, H - 1]):
                cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(VIS_DIR, f"result_{img_name}"), orig)

        rows.append({
            "image": img_name, "mean_error": round(errors.mean(), 6),
            **{f"lm{i+1}_error": round(errors[i], 6) for i in range(NUM_LANDMARKS)},
        })

all_errors = np.array(all_errors)
print(f"\n  --- Overall ({len(all_errors)} images) ---")
print(f"  Average error : {all_errors.mean():.6f}")
print(f"  Median error  : {np.median(all_errors):.6f}")
print(f"  Std deviation : {all_errors.std():.6f}")
print(f"  Min / Max     : {all_errors.min():.6f} / {all_errors.max():.6f}")

check(f"Average error below threshold ({ERROR_THRESHOLD})",
      all_errors.mean() < ERROR_THRESHOLD, f"Got {all_errors.mean():.6f}")
check("No NaN errors", not any(np.isnan(e) for e in all_errors))

csv_path   = os.path.join(OUTPUT_DIR, "accuracy_results.csv")
fieldnames = ["image", "mean_error"] + [f"lm{i+1}_error" for i in range(NUM_LANDMARKS)]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"\n  Results saved to {csv_path}")
print(f"  Visualizations saved to {VIS_DIR}/")
print(f"\n======== Summary ========")
print(f"  PASSED: {PASSED}")
print(f"  FAILED: {FAILED}")