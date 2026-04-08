import yaml
import os
import sys
import numpy as np
import torch
import cv2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
with open("../config.yaml") as f:
    config = yaml.safe_load(f)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adult_model import get_model, soft_argmax_2d

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

print("\n  --- Edge case Images ---")
edge_cases = [
    ("All-Black Image",  np.zeros((400, 400, 3), dtype=np.uint8)),
    ("All-White Image",  np.full((400, 400, 3), 255, dtype=np.uint8)),
]

for label, img in edge_cases:
    print(f"\n  --- {label} ---")
    try:
        pts = run_inference(model, img, (50, 50, 350, 350))
        check(f"Handles {label} without crashing", pts.shape == (NUM_LANDMARKS, 2))
        check(f"{label}: output coords in [0,1]",  pts.min() >= 0.0 and pts.max() <= 1.0)
    except Exception as e:
        check(f"Handles {label} without crashing", False, str(e))


print("\n  --- Wide Aspect Ratio Bbox (2:1) ---")
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