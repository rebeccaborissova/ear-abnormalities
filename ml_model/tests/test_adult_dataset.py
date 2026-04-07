import os
import yaml
import sys
import csv
import cv2
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
with open("../config.yaml") as f:
    config = yaml.safe_load(f)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adult_model import get_model, soft_argmax_2d

BASE_DIR = "/home/UFAD/jingyifu/ear-project/ear-abnormalities/resnet-18/model"
CKPT_23          = os.path.join(BASE_DIR, "infant_ear_model_23lm_best.pth")
ADULT_DIR        = os.environ.get("ADULT_DIR", "")
OUTPUT_DIR       = "test_outputs/adult_dataset"
VIS_DIR          = "test_outputs/adult_dataset/visualizations"
NUM_LANDMARKS    = 23
NUM_STAGES       = 6
INPUT_SIZE       = 368
NUM_IMAGES       = 20
BBOX_PADDING     = 20
ERROR_THRESHOLD  = 0.35
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

# 55→22 mapping
INDICES_55_TO_22 = list(range(20)) + [37, 25]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR,    exist_ok=True)

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
                inside = True; continue
            if line == "}":
                break
            if inside:
                x, y = line.split()
                pts.append([float(x), float(y)])
    except Exception:
        return None
    return np.array(pts) if len(pts) == 55 else None


def preprocess(img_bgr, x1, y1, x2, y2):
    crop    = cv2.resize(img_bgr[y1:y2, x1:x2], (INPUT_SIZE, INPUT_SIZE))
    rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x_t     = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    return (x_t - MEAN) / STD

print("\n======== 23-Point Model on Adult Dataset ========")
model = get_model(NUM_LANDMARKS, NUM_STAGES).to(DEVICE)
if os.path.exists(CKPT_23):
    model.load_state_dict(torch.load(CKPT_23, map_location=DEVICE))
    print(f"  Loaded {CKPT_23}")
else:
    print(f"  WARNING: {CKPT_23} not found, using random weights")
model.eval()

# collect pairs
pairs = []
for fname in sorted(os.listdir(ADULT_DIR)):
    if not fname.endswith(".png"):
        continue
    txt = os.path.join(ADULT_DIR, fname.replace(".png", ".txt"))
    if os.path.exists(txt):
        pairs.append((os.path.join(ADULT_DIR, fname), txt, fname))
    if len(pairs) >= NUM_IMAGES:
        break

print(f"  Found {len(pairs)} adult ear image/label pairs")

all_errors = []
rows       = []

for img_path, txt_path, fname in pairs:
    img = cv2.imread(img_path)
    if img is None:
        continue
    H, W = img.shape[:2]

    pts55 = read_55pt_labels(txt_path)
    if pts55 is None:
        continue

    # bounding box from landmarks
    x1 = max(0, int(pts55[:, 0].min()) - BBOX_PADDING)
    y1 = max(0, int(pts55[:, 1].min()) - BBOX_PADDING)
    x2 = min(W, int(pts55[:, 0].max()) + BBOX_PADDING)
    y2 = min(H, int(pts55[:, 1].max()) + BBOX_PADDING)
    bw, bh = x2 - x1, y2 - y1
    if bw < 5 or bh < 5:
        continue

    # ground truth (22 mapped landmarks, normalised to crop)
    gt22     = pts55[INDICES_55_TO_22]
    gt22_n   = np.clip(
        np.stack([(gt22[:, 0] - x1) / (bw - 1), (gt22[:, 1] - y1) / (bh - 1)], axis=1),
        0.0, 1.0,
    )

    with torch.no_grad():
        out  = model(preprocess(img, x1, y1, x2, y2))
        heat = out[:, -1] if out.ndim == 5 else out
        pred = soft_argmax_2d(heat, normalize=True)[0].cpu().numpy()

    errors   = np.sqrt(np.sum((pred[:22] - gt22_n) ** 2, axis=1))
    mean_err = errors.mean()
    all_errors.append(mean_err)

    # visualize
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    for x, y in gt22:                                              # green = GT
        cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
    for x, y in pred * np.array([bw - 1, bh - 1]) + np.array([x1, y1]):  # red = pred
        cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(VIS_DIR, f"result_{fname}"), vis)

    rows.append({"image": fname, "mean_error": round(mean_err, 6)})

all_errors = np.array(all_errors)
print(f"\n  --- Overall ({len(all_errors)} adult images) ---")
print(f"  Average error : {all_errors.mean():.6f}")
print(f"  Median error  : {np.median(all_errors):.6f}")
print(f"  Std deviation : {all_errors.std():.6f}")
print(f"  Min / Max     : {all_errors.min():.6f} / {all_errors.max():.6f}")

check(
    f"Average error below threshold ({ERROR_THRESHOLD})",
    all_errors.mean() < ERROR_THRESHOLD,
    f"Got {all_errors.mean():.6f}",
)
check("No NaN errors", not any(np.isnan(e) for e in all_errors))

csv_path = os.path.join(OUTPUT_DIR, "adult_accuracy_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image", "mean_error"])
    w.writeheader()
    w.writerows(rows)
print(f"\n  Results saved to {csv_path}")
print(f"  Visualizations saved to {VIS_DIR}/")

print(f"\n======== Summary ========")
print(f"  PASSED: {PASSED}")
print(f"  FAILED: {FAILED}")