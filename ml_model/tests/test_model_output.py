import os
import sys
import numpy as np
import torch
import cv2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# import from ml_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adult_model import get_model, soft_argmax_2d

BASE_DIR = "/home/UFAD/jingyifu/ear-project/ear-abnormalities/resnet-18/model"
CKPT_55      = os.path.join(BASE_DIR, "ear_landmark_model_best.pth")
CKPT_23      = os.path.join(BASE_DIR, "infant_ear_model_23lm_best.pth")
BOUNDING_CSV = os.environ.get("INFANT_BOUNDING_CSV", "")
IMAGES_DIR   = os.environ.get("INFANT_IMAGES_DIR", "")
INPUT_SIZE   = 368
NUM_STAGES   = 6
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

# 55→22 landmark mapping
LANDMARK_MAPPING = {i: i for i in range(20)}
LANDMARK_MAPPING.update({20: 37, 21: 25})
INDICES_55_TO_22 = [LANDMARK_MAPPING[i] for i in range(22)]

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


# load a real image
real_input = None
if BOUNDING_CSV and IMAGES_DIR:
    df = pd.read_csv(BOUNDING_CSV)
    for _, row in df.iterrows():
        img_name = os.path.basename(str(row["image_pathname"]).strip())
        img = cv2.imread(os.path.join(IMAGES_DIR, img_name))
        if img is None:
            continue
        H, W = img.shape[:2]
        x1 = max(0, int(row["left_bound"]))
        y1 = max(0, int(row["top_bound"]))
        x2 = min(W, int(row["right_bound"]))
        y2 = min(H, int(row["bottom_bound"]))
        crop    = cv2.resize(img[y1:y2, x1:x2], (INPUT_SIZE, INPUT_SIZE))
        rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        x_t     = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        real_input = (x_t - MEAN) / STD
        print(f"  Using image: {img_name}")
        break

if real_input is None:
    real_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)


# 55-Point Model
print("\n======== 55-Point Model Output Tests ========")
model_55 = get_model(55, NUM_STAGES).to(DEVICE)
if os.path.exists(CKPT_55):
    model_55.load_state_dict(torch.load(CKPT_55, map_location=DEVICE))
    print(f"  Loaded {CKPT_55}")
else:
    print(f"  WARNING: {CKPT_55} not found, using random weights")
model_55.eval()

with torch.no_grad():
    out   = model_55(real_input)
    heat  = out[:, -1] if out.ndim == 5 else out
    pts55 = soft_argmax_2d(heat, normalize=True)[0].cpu().numpy()

check("Outputs 55 landmark channels",       heat.shape[1] == 55,              f"Got {heat.shape[1]}")
check("Valid heatmap spatial dims",          heat.shape[2] > 0 and heat.shape[3] > 0)
check("No NaN in output",                   not torch.isnan(heat).any().item())
check("No Inf in output",                   not torch.isinf(heat).any().item())
check("soft_argmax shape is (55,2)",         pts55.shape == (55, 2),           f"Got {pts55.shape}")
check("All coords in [0,1]",                pts55.min() >= 0.0 and pts55.max() <= 1.0)


# 55→22 Mapping
print("\n======== 55→22 Landmark Mapping Tests ========")
check("Mapping covers exactly 22 landmarks",     len(INDICES_55_TO_22) == 22)
check("All source indices valid (0-54)",         all(0 <= i < 55 for i in INDICES_55_TO_22))
check("No duplicate source indices",             len(set(INDICES_55_TO_22)) == 22)

pts22 = pts55[INDICES_55_TO_22]
check("Mapped coords in [0,1]",                  pts22.min() >= 0.0 and pts22.max() <= 1.0)


# 23-point infant model
print("\n======== 23-Point Model Output Tests ========")
model_23 = get_model(23, NUM_STAGES).to(DEVICE)
if os.path.exists(CKPT_23):
    model_23.load_state_dict(torch.load(CKPT_23, map_location=DEVICE))
    print(f"  Loaded {CKPT_23}")
else:
    print(f"  WARNING: {CKPT_23} not found, using random weights")
model_23.eval()

with torch.no_grad():
    out   = model_23(real_input)
    heat  = out[:, -1] if out.ndim == 5 else out
    pts23 = soft_argmax_2d(heat, normalize=True)[0].cpu().numpy()

check("Outputs 23 landmark channels",   heat.shape[1] == 23,  f"Got {heat.shape[1]}")
check("No NaN in output",               not torch.isnan(heat).any().item())
check("No Inf in output",               not torch.isinf(heat).any().item())
check("soft_argmax shape is (23,2)",    pts23.shape == (23, 2))
check("All coords in [0,1]",           pts23.min() >= 0.0 and pts23.max() <= 1.0)

print(f"\n======== Summary ========")
print(f"  PASSED: {PASSED}")
print(f"  FAILED: {FAILED}")