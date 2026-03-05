import os
import sys
import numpy as np
import torch
import cv2
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import get_model, soft_argmax_2d

# config
CKPT_55 = "../ear_landmark_model_best.pth"
CKPT_23 = "../infant_ear_model_23lm_best.pth"
NUM_STAGES = 6
INPUT_SIZE = 368
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_DIR = os.environ.get("INFANT_IMAGES_DIR", "")
BOUNDING_CSV = os.environ.get("INFANT_BOUNDING_CSV", "")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

# 55→22 landmark mapping
LANDMARK_MAPPING_55_TO_22 = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 37,
    21: 25,
}
INDICES_55_TO_22 = [LANDMARK_MAPPING_55_TO_22[i] for i in range(22)]

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


real_input = None
if BOUNDING_CSV and IMAGES_DIR:
    df = pd.read_csv(BOUNDING_CSV)
    for _, row in df.iterrows():
        img_name = os.path.basename(str(row["image_pathname"]).strip())
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        x1 = max(0, int(row["left_bound"]))
        y1 = max(0, int(row["top_bound"]))
        x2 = int(row["right_bound"])
        y2 = int(row["bottom_bound"])
        crop = img[y1:y2, x1:x2]
        resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x_t = (
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            / 255.0
        )
        real_input = (x_t - MEAN) / STD
        print(f"Image: {img_name}")
        break

if real_input is None:
    print("  WARNING: no real image found, falling back to dummy input")
    real_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

# Verifies that the original 55-landmark ResNet-18 model applies
print("\n======== 55-Point Model Output Tests ========")
model_55 = get_model(55, NUM_STAGES).to(DEVICE)
if os.path.exists(CKPT_55):
    model_55.load_state_dict(torch.load(CKPT_55, map_location=DEVICE))
    print(f"  Loaded {CKPT_55}")
else:
    print(f"  WARNING: {CKPT_55} not found, using random weights")
model_55.eval()

with torch.no_grad():
    out_55 = model_55(real_input)
    if isinstance(out_55, (list, tuple)):
        heat_55 = out_55[-1]
    elif out_55.ndim == 5:
        heat_55 = out_55[:, -1]
    else:
        heat_55 = out_55

    # Verify model outputs correct number of landmark channels
    check(
        "55-point model outputs 55 landmark channels",
        heat_55.shape[1] == 55,
        f"Got {heat_55.shape[1]}",
    )

    # Verify heatmap has valid spatial dimensions
    check(
        "Heatmap spatial dims are positive",
        heat_55.shape[2] > 0 and heat_55.shape[3] > 0,
        f"Shape: {heat_55.shape}",
    )

    # Verify no NaN values in output
    check(
        "No NaN in 55-point heatmap output",
        not torch.isnan(heat_55).any().item(),
        "NaN values detected",
    )

    # Verify no Inf values in output
    check(
        "No Inf in 55-point heatmap output",
        not torch.isinf(heat_55).any().item(),
        "Inf values detected",
    )

    pts_55 = soft_argmax_2d(heat_55, normalize=True)[0].cpu().numpy()

    # Verify soft_argmax returns correct shape
    check(
        "soft_argmax output shape is (55,2)",
        pts_55.shape == (55, 2),
        f"Got {pts_55.shape}",
    )

    # Verify all predicted coordinates are within valid [0,1] range
    check(
        "All 55 predicted coords in [0,1]",
        pts_55.min() >= 0.0 and pts_55.max() <= 1.0,
        f"Range: [{pts_55.min():.4f}, {pts_55.max():.4f}]",
    )


# Verifies the landmark mapping is successful
print("\n======== 55→22 Mapping Tests ========")

check("Mapping covers exactly 22 new landmarks", len(INDICES_55_TO_22) == 22)

check(
    "All source indices are valid (0-54)",
    all(0 <= i < 55 for i in INDICES_55_TO_22),
    f"Invalid indices: {[i for i in INDICES_55_TO_22 if not (0<=i<55)]}",
)

check(
    "No duplicate source indices in mapping",
    len(set(INDICES_55_TO_22)) == len(INDICES_55_TO_22),
    f"Duplicates found",
)

pts_22 = pts_55[INDICES_55_TO_22]
check(
    "Mapped 22 landmarks have correct shape",
    pts_22.shape == (22, 2),
    f"Got {pts_22.shape}",
)

check(
    "Mapped 22 landmarks are within [0,1]", pts_22.min() >= 0.0 and pts_22.max() <= 1.0
)

# Verify 23-point model output shape applies
print("\n======== 23-Point Model Output Tests ========")
model_23 = get_model(23, NUM_STAGES).to(DEVICE)
if os.path.exists(CKPT_23):
    model_23.load_state_dict(torch.load(CKPT_23, map_location=DEVICE))
    print(f"  Loaded {CKPT_23}")
else:
    print(f"  WARNING: {CKPT_23} not found, using random weights")
model_23.eval()

with torch.no_grad():
    out_23 = model_23(real_input)
    if isinstance(out_23, (list, tuple)):
        heat_23 = out_23[-1]
    elif out_23.ndim == 5:
        heat_23 = out_23[:, -1]
    else:
        heat_23 = out_23

    check(
        "23-point model outputs exactly 23 landmark channels",
        heat_23.shape[1] == 23,
        f"Got {heat_23.shape[1]}",
    )

    check("No NaN in 23-point heatmap output", not torch.isnan(heat_23).any().item())

    check("No Inf in 23-point heatmap output", not torch.isinf(heat_23).any().item())

    pts_23 = soft_argmax_2d(heat_23, normalize=True)[0].cpu().numpy()

    check(
        "Final output contains exactly 23 landmarks",
        pts_23.shape[0] == 23,
        f"Got {pts_23.shape[0]}",
    )

    check(
        "Final 23 landmarks are in correct order (shape (23,2))",
        pts_23.shape == (23, 2),
    )

    spread_23 = pts_23.std(axis=0).mean()
    check(
        "23 landmarks are spread across image (std > 0.01)",
        spread_23 > 0.01,
        f"Std={spread_23:.4f} — points may be collapsed",
    )

# Summary
print(f"\n======== Summary ========")
print(f"  PASSED: {PASSED}")
print(f"  FAILED: {FAILED}")
