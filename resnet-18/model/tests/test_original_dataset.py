import os
import sys
import cv2
import numpy as np
import torch
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import get_model, soft_argmax_2d
from infant_dataset import get_train_test_split

# config
CKPT_55 = "../ear_landmark_model_best.pth"
CKPT_23 = "../infant_ear_model_23lm_best.pth"
NUM_STAGES = 6
OUTPUT_DIR = "test_outputs/visualization_55"
IMAGES_DIR = os.environ.get("INFANT_IMAGES_DIR", "")
NUM_IMAGES = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 55->22 mapping
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
MAPPED_55_INDICES = [LANDMARK_MAPPING_55_TO_22[i] for i in range(22)]
HIGHLIGHTED_55 = set(MAPPED_55_INDICES)


def get_rainbow_colors(n):
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


COLORS = get_rainbow_colors(23)


def hard_argmax_2d(heat):
    B, C, H, W = heat.shape
    flat = heat.view(B, C, -1)
    idx_flat = flat.argmax(dim=-1)
    y = (idx_flat // W).float() / (H - 1)
    x = (idx_flat % W).float() / (W - 1)
    return torch.stack([x, y], dim=-1)[0]


# load models
print(f"\nLoading models...")
model_55 = get_model(55, NUM_STAGES).to(DEVICE)
model_55.load_state_dict(torch.load(CKPT_55, map_location=DEVICE))
model_55.eval()

model_23 = get_model(23, NUM_STAGES).to(DEVICE)
model_23.load_state_dict(torch.load(CKPT_23, map_location=DEVICE))
model_23.eval()
print(f"Both models loaded. Device: {DEVICE}")

# load datasets
_, test_dataset = get_train_test_split(num_landmarks=23)
print(f"Saving {min(NUM_IMAGES, len(test_dataset))} visualizations to {OUTPUT_DIR}/\n")

# visualize
for idx in range(min(NUM_IMAGES, len(test_dataset))):
    img_t, _ = test_dataset[idx]
    img_name = test_dataset.image_files[idx]

    img_input = img_t.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # 55-point model
        out_55 = model_55(img_input)
        if isinstance(out_55, (list, tuple)):
            heat_55 = out_55[-1]
        elif out_55.ndim == 5:
            heat_55 = out_55[:, -1]
        else:
            heat_55 = out_55
        pts_55 = hard_argmax_2d(heat_55).cpu().numpy()

        # combined model - accurate landmark positions
        out_23 = model_23(img_input)
        if isinstance(out_23, (list, tuple)):
            heat_23 = out_23[-1]
        elif out_23.ndim == 5:
            heat_23 = out_23[:, -1]
        else:
            heat_23 = out_23
        pts_23 = soft_argmax_2d(heat_23, normalize=True)[0].cpu().numpy()

    orig = cv2.imread(os.path.join(IMAGES_DIR, img_name))
    if orig is None:
        print(f"  WARNING: could not read {img_name}, skipping")
        continue
    H, W = orig.shape[:2]

    pts_55_px = pts_55 * np.array([W - 1, H - 1])
    pts_23_px = pts_23 * np.array([W - 1, H - 1])

    vis = orig.copy()

    # draw 32 non-selected points
    for i, (px, py) in enumerate(pts_55_px):
        if i not in HIGHLIGHTED_55:
            cv2.circle(vis, (int(px), int(py)), 6, (0, 0, 0), -1)
            cv2.circle(vis, (int(px), int(py)), 4, (255, 255, 255), -1)

    # draw 23 accurate points from combined model
    for i in range(23):
        px, py = pts_23_px[i]
        color = COLORS[i]
        cv2.circle(vis, (int(px), int(py)), 7, (0, 0, 0), -1)
        cv2.circle(vis, (int(px), int(py)), 5, color, -1)
        cv2.putText(
            vis,
            str(i + 1),
            (int(px) + 7, int(py) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out_path = os.path.join(OUTPUT_DIR, f"vis_{img_name}")
    cv2.imwrite(out_path, vis)
    print(f"  [{idx+1}/{min(NUM_IMAGES, len(test_dataset))}] Saved: vis_{img_name}")
