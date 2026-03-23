import os
import sys
import csv
import cv2
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gradio.model import get_model, soft_argmax_2d

# config
CKPT_23 = "../infant_ear_model_23lm_best.pth"
NUM_LANDMARKS = 23
NUM_STAGES = 6
INPUT_SIZE = 368
ADULT_DIR = os.environ.get("ADULT_DIR", "")
OUTPUT_DIR = "test_outputs/adult_dataset"
VIS_DIR = "test_outputs/adult_dataset/visualizations"
NUM_IMAGES = 20
BBOX_PADDING = 20
ERROR_THRESHOLD_PASS = 0.35

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

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
INDICES_55 = [LANDMARK_MAPPING_55_TO_22[i] for i in range(22)]

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


def read_55pt_labels(txt_path):
    pts = []
    try:
        with open(txt_path) as f:
            lines = f.readlines()
        inside = False
        for line in lines:
            line = line.strip()
            if line == "{":
                inside = True
                continue
            if line == "}":
                break
            if inside:
                x, y = line.split()
                pts.append([float(x), float(y)])
    except Exception:
        return None
    if len(pts) != 55:
        return None
    return np.array(pts)


def bbox_from_pts(pts, H, W, padding=20):
    x1 = max(0, int(pts[:, 0].min()) - padding)
    y1 = max(0, int(pts[:, 1].min()) - padding)
    x2 = min(W, int(pts[:, 0].max()) + padding)
    y2 = min(H, int(pts[:, 1].max()) + padding)
    return x1, y1, x2, y2


def preprocess_crop(img_bgr, bbox):
    x1, y1, x2, y2 = bbox
    crop = img_bgr[y1:y2, x1:x2]
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    return (x_t - MEAN) / STD


# load model
model_23 = get_model(NUM_LANDMARKS, NUM_STAGES).to(DEVICE)
if os.path.exists(CKPT_23):
    model_23.load_state_dict(torch.load(CKPT_23, map_location=DEVICE))
    print(f"  Loaded {CKPT_23}")
else:
    print(f"  WARNING: {CKPT_23} not found")
model_23.eval()

pairs = []
for fname in sorted(os.listdir(ADULT_DIR)):
    if not fname.endswith(".png"):
        continue
    txt = os.path.join(ADULT_DIR, fname.replace(".png", ".txt"))
    img_p = os.path.join(ADULT_DIR, fname)
    if os.path.exists(txt):
        pairs.append((img_p, txt, fname))
    if len(pairs) >= NUM_IMAGES:
        break

print(f"  Found {len(pairs)} adult ear image/label pairs")

all_errors = []
rows = []

for img_path, txt_path, fname in pairs:
    img = cv2.imread(img_path)
    if img is None:
        continue
    H, W = img.shape[:2]

    pts_55 = read_55pt_labels(txt_path)
    if pts_55 is None:
        continue

    x1, y1, x2, y2 = bbox_from_pts(pts_55, H, W, padding=BBOX_PADDING)
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 5:
        continue

    gt_22_px = pts_55[INDICES_55]
    gt_22_norm = np.zeros_like(gt_22_px)
    gt_22_norm[:, 0] = (gt_22_px[:, 0] - x1) / (bw - 1)
    gt_22_norm[:, 1] = (gt_22_px[:, 1] - y1) / (bh - 1)
    gt_22_norm = np.clip(gt_22_norm, 0.0, 1.0)

    x_t = preprocess_crop(img, (x1, y1, x2, y2))
    with torch.no_grad():
        out = model_23(x_t)
        if isinstance(out, (list, tuple)):
            heat = out[-1]
        elif out.ndim == 5:
            heat = out[:, -1]
        else:
            heat = out
        pts_23_norm = soft_argmax_2d(heat, normalize=True)[0].cpu().numpy()

    pred_22_norm = pts_23_norm[:22]
    errors = np.sqrt(np.sum((pred_22_norm - gt_22_norm) ** 2, axis=1))
    mean_err = errors.mean()
    all_errors.append(mean_err)

    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    pred_px = pts_23_norm * np.array([bw - 1, bh - 1]) + np.array([x1, y1])
    gt_px = gt_22_px

    # ground truth: green
    for x, y in gt_px:
        cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)

    # predicted: red
    for x, y in pred_px:
        cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(VIS_DIR, f"result_{fname}"), vis)

    rows.append(
        {
            "image": fname,
            "mean_error": round(mean_err, 6),
            **{f"lm{i+1}_error": round(errors[i], 6) for i in range(22)},
        }
    )

all_errors = np.array(all_errors)

print(f"\n  --- Overall ({len(all_errors)} adult images) ---")
print(f"  Average error : {all_errors.mean():.6f}")
print(f"  Median error  : {np.median(all_errors):.6f}")
print(f"  Std deviation : {all_errors.std():.6f}")
print(f"  Min error     : {all_errors.min():.6f}")
print(f"  Max error     : {all_errors.max():.6f}")


# Verify model within threshold
check(
    f"Average error on adult ears below threshold ({ERROR_THRESHOLD_PASS})",
    all_errors.mean() < ERROR_THRESHOLD_PASS,
    f"Got {all_errors.mean():.6f}",
)

csv_path = os.path.join(OUTPUT_DIR, "adult_accuracy_results.csv")
fieldnames = ["image", "mean_error"] + [f"lm{i+1}_error" for i in range(22)]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"\n  Results saved to {csv_path}")
print(f"  Visualizations saved to {VIS_DIR}/")

# Summary
print(f"\n======== Summary ========")
print(f"  PASSED: {PASSED}")
print(f"  FAILED: {FAILED}")
