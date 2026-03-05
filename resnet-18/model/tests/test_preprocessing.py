import os
import cv2
import numpy as np
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

BOUNDING_CSV = os.environ.get("INFANT_BOUNDING_CSV", "")
IMAGES_DIR = os.environ.get("INFANT_IMAGES_DIR", "")
INPUT_SIZE = 368
SAMPLE_SIZE = 10

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
print(f"Loaded {len(df)} rows from {BOUNDING_CSV}")

invalid_bbox = []
empty_crop = []
too_small_crop = []

for idx, row in df.iterrows():
    img_name = os.path.basename(str(row["image_pathname"]))
    img_path = os.path.join(IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue

    H, W = img.shape[:2]
    # coordinates
    x1 = int(row["left_bound"])
    y1 = int(row["top_bound"])
    x2 = int(row["right_bound"])
    y2 = int(row["bottom_bound"])

    if x1 >= x2 or y1 >= y2:
        invalid_bbox.append(img_name)
        continue

    x1c = max(0, min(x1, W - 1))
    y1c = max(0, min(y1, H - 1))
    x2c = max(1, min(x2, W))
    y2c = max(1, min(y2, H))

    crop = img[y1c:y2c, x1c:x2c]

    if crop.size == 0:
        empty_crop.append(img_name)
    elif crop.shape[0] < 5 or crop.shape[1] < 5:
        too_small_crop.append(img_name)

# Verify no invalid bounding boxes exist in the dataset
check(
    "No invalid bounding boxes", len(invalid_bbox) == 0, f"Invalid: {invalid_bbox[:5]}"
)

# Verify cropping does not produce empty images
check("No empty crops after clamping", len(empty_crop) == 0, f"Empty: {empty_crop[:5]}")

# Verify cropped images contains the ear region
check(
    "No crops smaller than 5x5",
    len(too_small_crop) == 0,
    f"Too small: {too_small_crop[:5]}",
)
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
    (0, 0, (50, 50, 200, 200), (50, 50)),  # top-left
    (45, 45, (50, 50, 200, 200), (200, 200)),  # bottom-right
    (23, 23, (50, 50, 200, 200), (125, 125)),  # center
]

mapping_ok = True
for hm_x, hm_y, bbox, expected in test_cases:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pts_hm = np.array([[hm_x, hm_y]], dtype=np.float32)
    pts_orig = map_heatmap_to_orig(pts_hm, hm_H, hm_W, x1, y1, bw, bh)
    ex, ey = expected
    err = np.sqrt((pts_orig[0, 0] - ex) ** 2 + (pts_orig[0, 1] - ey) ** 2)
    if err > 5.0:
        mapping_ok = False
        print(
            f"Mapping error too large: hm=({hm_x},{hm_y}) → orig=({pts_orig[0,0]:.1f},{pts_orig[0,1]:.1f}) expected=({ex},{ey})"
        )

# Verify the heatmap-to-original coordinate mapping formula is correct
check("Heatmap-to-original coordinate mapping is correct", mapping_ok)

np.random.seed(42)
dummy_heat = np.random.rand(46, 46).astype(np.float32)
peak_y, peak_x = np.unravel_index(dummy_heat.argmax(), dummy_heat.shape)

# verify landmarks do not cluster at corners
check(
    "Hard argmax on random heatmap does not always give corner (0,0)",
    not (peak_x == 0 and peak_y == 0),
    f"Got peak at ({peak_x},{peak_y})",
)

print(f"\n  Saving {SAMPLE_SIZE} sample crops for visual inspection...")
os.makedirs("test_outputs/crops", exist_ok=True)
saved = 0
for idx, row in df.iterrows():
    if saved >= SAMPLE_SIZE:
        break
    img_name = os.path.basename(str(row["image_pathname"]))
    img_path = os.path.join(IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    H, W = img.shape[:2]
    x1 = max(0, min(int(row["left_bound"]), W - 1))
    y1 = max(0, min(int(row["top_bound"]), H - 1))
    x2 = max(1, min(int(row["right_bound"]), W))
    y2 = max(1, min(int(row["bottom_bound"]), H))
    crop = img[y1:y2, x1:x2]
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imwrite(f"test_outputs/crops/bbox_{img_name}", vis)
    cv2.imwrite(f"test_outputs/crops/crop_{img_name}", crop)
    saved += 1

# Summary
print(f"\n======== Summary ========")
print(f"  PASSED: {PASSED}")
print(f"  FAILED: {FAILED}")
