import yaml
import os
import sys
import numpy as np
import torch
import yaml
import cv2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
with open("../config.yaml") as f:
    config = yaml.safe_load(f)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adult_model import get_model, soft_argmax_2d
from infant_dataset import get_train_test_split

# config
BASE_DIR = "/home/UFAD/jingyifu/ear-project/ear-abnormalities/resnet-18/model"
CKPT_23       = os.path.join(BASE_DIR, "infant_ear_model_23lm_best.pth")
BOUNDING_CSV  = os.environ.get("INFANT_BOUNDING_CSV", "")
IMAGES_DIR    = os.environ.get("INFANT_IMAGES_DIR", "")
OUTPUT_DIR    = "test_outputs/robustness"
INPUT_SIZE    = 368
NUM_LANDMARKS = 23
NUM_STAGES    = 6
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def run_inference(model, img_bgr, bbox):
    x1, y1, x2, y2 = bbox
    crop    = cv2.resize(img_bgr[y1:y2, x1:x2], (INPUT_SIZE, INPUT_SIZE))
    rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x_t     = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    x_t     = (x_t - MEAN) / STD
    with torch.no_grad():
        out  = model(x_t)
        heat = out[:, -1] if out.ndim == 5 else out
        return soft_argmax_2d(heat, normalize=True)[0].cpu().numpy()


# load model
print("\n======== Robustness Testing ========")
model = get_model(NUM_LANDMARKS, NUM_STAGES).to(DEVICE)
if os.path.exists(CKPT_23):
    model.load_state_dict(torch.load(CKPT_23, map_location=DEVICE))
    print(f"  Loaded {CKPT_23}")
else:
    print(f"  WARNING: {CKPT_23} not found, using random weights")
model.eval()


print("\n  --- Test 1: Real Ear Images ---")
_, test_dataset = get_train_test_split(config, num_landmarks=NUM_LANDMARKS)
spreads = []

for idx in range(min(5, len(test_dataset))):
    img_t, _ = test_dataset[idx]
    img_name  = test_dataset.image_files[idx]

    with torch.no_grad():
        out  = model(img_t.unsqueeze(0).to(DEVICE))
        heat = out[:, -1] if out.ndim == 5 else out
        pts  = soft_argmax_2d(heat, normalize=True)[0].cpu().numpy()

    orig = cv2.imread(os.path.join(IMAGES_DIR, img_name))
    if orig is not None:
        H, W = orig.shape[:2]
        for x, y in pts * np.array([W - 1, H - 1]):
            cv2.circle(orig, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"real_{img_name}"), orig)

    spreads.append(pts.std(axis=0).mean())

if spreads:
    avg_spread = np.mean(spreads)
    print(f"  Average landmark spread: {avg_spread:.4f}")
    check(
        "Landmarks spread across ear (std > 0.01)",
        avg_spread > 0.01,
        f"Spread={avg_spread:.4f}",
    )
    check(
        "All predicted coords in [0,1]",
        all(pts.min() >= 0.0 and pts.max() <= 1.0 for _ in [1]),
    )


print("\n  --- Test 2: Small Bounding Box (50×50) ---")
dummy = np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
try:
    pts = run_inference(model, dummy, (125, 125, 175, 175))
    check("Handles 50×50 bbox without crashing",   pts.shape == (NUM_LANDMARKS, 2))
    check("Small bbox: output coords in [0,1]",    pts.min() >= 0.0 and pts.max() <= 1.0)
except Exception as e:
    check("Handles 50×50 bbox without crashing", False, str(e))


print("\n  --- Test 3: All-Black Image ---")
try:
    pts = run_inference(model, np.zeros((400, 400, 3), dtype=np.uint8), (50, 50, 350, 350))
    check("Handles all-black image without crashing", pts.shape == (NUM_LANDMARKS, 2))
    check("Black image: output coords in [0,1]",      pts.min() >= 0.0 and pts.max() <= 1.0)
except Exception as e:
    check("Handles all-black image without crashing", False, str(e))


print("\n  --- Test 4: All-White Image ---")
try:
    pts = run_inference(model, np.full((400, 400, 3), 255, dtype=np.uint8), (50, 50, 350, 350))
    check("Handles all-white image without crashing", pts.shape == (NUM_LANDMARKS, 2))
    check("White image: output coords in [0,1]",      pts.min() >= 0.0 and pts.max() <= 1.0)
except Exception as e:
    check("Handles all-white image without crashing", False, str(e))


print("\n  --- Test 5: Wide Aspect Ratio Bbox (2:1) ---")
try:
    pts = run_inference(model, np.random.randint(50, 200, (300, 600, 3), dtype=np.uint8), (50, 50, 550, 250))
    check("Handles 2:1 aspect ratio bbox without crashing", pts.shape == (NUM_LANDMARKS, 2))
except Exception as e:
    check("Handles 2:1 aspect ratio bbox without crashing", False, str(e))

# summary
print(f"\n======== Summary ========")
print(f"  PASSED: {PASSED}")
print(f"  FAILED: {FAILED}")
print(f"  Visualizations saved to {OUTPUT_DIR}/")