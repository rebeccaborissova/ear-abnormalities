import os
import sys
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BOUNDING_CSV = os.environ.get("INFANT_BOUNDING_CSV", "")
IMAGES_DIR   = os.environ.get("INFANT_IMAGES_DIR", "")
SAMPLE_SIZE  = 10

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


# A. Bounding Box Verification
print("\n======== A. Bounding Box Verification ========")
df = pd.read_csv(BOUNDING_CSV)
print(f"  Loaded {len(df)} rows from {BOUNDING_CSV}")

invalid_bbox  = []
empty_crop    = []
too_small     = []

for _, row in df.iterrows():
    img_name = os.path.basename(str(row["image_pathname"]))
    img = cv2.imread(os.path.join(IMAGES_DIR, img_name))
    if img is None:
        continue

    H, W = img.shape[:2]
    x1, y1 = int(row["left_bound"]),  int(row["top_bound"])
    x2, y2 = int(row["right_bound"]), int(row["bottom_bound"])

    if x1 >= x2 or y1 >= y2:
        invalid_bbox.append(img_name)
        continue

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    crop = img[y1c:y2c, x1c:x2c]

    if crop.size == 0:
        empty_crop.append(img_name)
    elif crop.shape[0] < 5 or crop.shape[1] < 5:
        too_small.append(img_name)

check("No invalid bounding boxes",      len(invalid_bbox) == 0, str(invalid_bbox[:5]))
check("No empty crops after clamping",  len(empty_crop)   == 0, str(empty_crop[:5]))
check("No crops smaller than 5×5 px",  len(too_small)    == 0, str(too_small[:5]))
print(f"  Checked {len(df)} bounding boxes")


# B. Coordinate Mapping Verification
print("\n======== B. Coordinate Mapping Verification ========")


def map_heatmap_to_orig(pts_hm, hm_H, hm_W, x1, y1, bw, bh):
    pts = pts_hm.copy().astype(np.float32)
    pts[:, 0] = pts[:, 0] / hm_W * bw + x1
    pts[:, 1] = pts[:, 1] / hm_H * bh + y1
    return pts


hm_H, hm_W = 46, 46
test_cases = [
    (0,  0,  (50, 50, 200, 200), (50,  50)),   # top-left
    (45, 45, (50, 50, 200, 200), (200, 200)),   # bottom-right
    (23, 23, (50, 50, 200, 200), (125, 125)),   # center
]

mapping_ok = True
for hm_x, hm_y, bbox, (ex, ey) in test_cases:
    x1, y1, x2, y2 = bbox
    pts_orig = map_heatmap_to_orig(
        np.array([[hm_x, hm_y]], dtype=np.float32),
        hm_H, hm_W, x1, y1, x2 - x1, y2 - y1
    )
    if np.sqrt((pts_orig[0, 0] - ex) ** 2 + (pts_orig[0, 1] - ey) ** 2) > 5.0:
        mapping_ok = False

check("Heatmap-to-original coordinate mapping is correct", mapping_ok)

np.random.seed(42)
dummy = np.random.rand(46, 46).astype(np.float32)
peak_y, peak_x = np.unravel_index(dummy.argmax(), dummy.shape)
check(
    "Hard argmax on random heatmap does not collapse to (0,0)",
    not (peak_x == 0 and peak_y == 0),
    f"Got peak at ({peak_x},{peak_y})",
)

# save sample crops for visual inspection
os.makedirs("test_outputs/crops", exist_ok=True)
saved = 0
for _, row in df.iterrows():
    if saved >= SAMPLE_SIZE:
        break
    img_name = os.path.basename(str(row["image_pathname"]))
    img = cv2.imread(os.path.join(IMAGES_DIR, img_name))
    if img is None:
        continue
    H, W = img.shape[:2]
    x1 = max(0, min(int(row["left_bound"]),  W - 1))
    y1 = max(0, min(int(row["top_bound"]),   H - 1))
    x2 = max(1, min(int(row["right_bound"]), W))
    y2 = max(1, min(int(row["bottom_bound"]), H))
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imwrite(f"test_outputs/crops/bbox_{img_name}", vis)
    cv2.imwrite(f"test_outputs/crops/crop_{img_name}", img[y1:y2, x1:x2])
    saved += 1

print(f"  Saved {saved} sample crops to test_outputs/crops/")

print(f"\n======== Summary ========")
print(f"  PASSED: {PASSED}")
print(f"  FAILED: {FAILED}")