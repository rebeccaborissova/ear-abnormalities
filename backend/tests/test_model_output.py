import os
import sys
import numpy as np
import torch
import cv2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adult_model import get_model, soft_argmax_2d

BASE_DIR     = "/home/UFAD/jingyifu/ear-project/ear-abnormalities/resnet-18/model"
CKPT_55      = os.path.join(BASE_DIR, "ear_landmark_model_best.pth")
CKPT_23      = os.path.join(BASE_DIR, "infant_ear_model_23lm_best.pth")
BOUNDING_CSV = os.environ.get("INFANT_BOUNDING_CSV", "")
IMAGES_DIR   = os.environ.get("INFANT_IMAGES_DIR", "")
INPUT_SIZE   = 368
NUM_STAGES   = 6
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

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

def load_real_input():
    if not (BOUNDING_CSV and IMAGES_DIR):
        return None
    df = pd.read_csv(BOUNDING_CSV)
    for _, row in df.iterrows():
        img_name = os.path.basename(str(row["image_pathname"]).strip())
        img = cv2.imread(os.path.join(IMAGES_DIR, img_name))
        if img is None:
            continue
        H, W = img.shape[:2]
        x1, y1 = max(0, int(row["left_bound"])),  max(0, int(row["top_bound"]))
        x2, y2 = min(W, int(row["right_bound"])), min(H, int(row["bottom_bound"]))
        crop = cv2.resize(img[y1:y2, x1:x2], (INPUT_SIZE, INPUT_SIZE))
        x_t  = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        print(f"  Using image: {img_name}")
        return (x_t - MEAN) / STD
    return None

real_input = load_real_input()
if real_input is None:
    real_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

def test_model(num_landmarks, ckpt, label):
    print(f"\n======== {label} ========")
    model = get_model(num_landmarks, NUM_STAGES).to(DEVICE)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"  Loaded {ckpt}")
    else:
        print(f"  WARNING: {ckpt} not found, using random weights")
    model.eval()

    with torch.no_grad():
        out  = model(real_input)
        heat = out[:, -1] if out.ndim == 5 else out
        pts  = soft_argmax_2d(heat, normalize=True)[0].cpu().numpy()

    check(f"Outputs {num_landmarks} landmark channels", heat.shape[1] == num_landmarks, f"Got {heat.shape[1]}")
    check("Valid heatmap spatial dims",                 heat.shape[2] > 0 and heat.shape[3] > 0)
    check("No NaN in output",                          not torch.isnan(heat).any().item())
    check("No Inf in output",                          not torch.isinf(heat).any().item())
    check(f"soft_argmax shape is ({num_landmarks},2)",  pts.shape == (num_landmarks, 2))
    check("All coords in [0,1]",                       pts.min() >= 0.0 and pts.max() <= 1.0)

test_model(55, CKPT_55, "55-Point Model Output Tests")
test_model(23, CKPT_23, "23-Point Model Output Tests")

print(f"\n======== Summary ========")
print(f"  PASSED: {PASSED}")
print(f"  FAILED: {FAILED}")