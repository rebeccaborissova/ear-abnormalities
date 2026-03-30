import os
import sys
import numpy as np
import torch
import cv2
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gradio.model import get_model, soft_argmax_2d

# config
NUM_LANDMARKS = 23
NUM_STAGES = 6
CKPT_23 = "../infant_ear_model_23lm_best.pth"
BOUNDING_CSV = os.environ.get("INFANT_BOUNDING_CSV", "")
IMAGES_DIR = os.environ.get("INFANT_IMAGES_DIR", "")
INPUT_SIZE = 368
OUTPUT_DIR = "test_outputs/robustness"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

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
    crop = img_bgr[y1:y2, x1:x2]
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    x_t = (x_t - MEAN) / STD
    with torch.no_grad():
        out = model(x_t)
        if isinstance(out, (list, tuple)):
            heat = out[-1]
        elif out.ndim == 5:
            heat = out[:, -1]
        else:
            heat = out
        pts = soft_argmax_2d(heat, normalize=True)[0].cpu().numpy()  # (23,2)
    return pts


# load model
print("\n======== Robustness Testing ========")
model_23 = get_model(NUM_LANDMARKS, NUM_STAGES).to(DEVICE)
if os.path.exists(CKPT_23):
    model_23.load_state_dict(torch.load(CKPT_23, map_location=DEVICE))
    print(f"  Loaded {CKPT_23}")
else:
    print(f"  WARNING: {CKPT_23} not found, using random weights")
model_23.eval()

# load images
real_images = []
if BOUNDING_CSV and IMAGES_DIR:
    df = pd.read_csv(BOUNDING_CSV)
    for _, row in df.iterrows():
        if len(real_images) >= 5:
            break
        img_name = os.path.basename(str(row["image_pathname"]).strip())
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
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        real_images.append((img_name, img, (x1, y1, x2, y2)))
    print(f"  Loaded {len(real_images)} real images for testing")

print("\n  --- Test 1: Real Ear Images ---")
from infant_dataset import get_train_test_split

_, test_dataset = get_train_test_split(num_landmarks=NUM_LANDMARKS)

spreads = []
for idx in range(min(5, len(test_dataset))):
    img, _ = test_dataset[idx]
    img_name = test_dataset.image_files[idx]

    img_t = img.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model_23(img_t)
        if isinstance(out, (list, tuple)):
            heat = out[-1]
        elif out.ndim == 5:
            heat = out[:, -1]
        else:
            heat = out
        pts = soft_argmax_2d(heat, normalize=True)[0].cpu().numpy()

    orig = cv2.imread(os.path.join(IMAGES_DIR, img_name))
    if orig is None:
        continue
    H, W = orig.shape[:2]
    pred_px = pts * np.array([W - 1, H - 1])
    for x, y in pred_px:
        cv2.circle(orig, (int(x), int(y)), 4, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"real_{img_name}"), orig)

    spread = pts.std(axis=0).mean()
    spreads.append(spread)

    orig = cv2.imread(os.path.join(IMAGES_DIR, img_name))
    if orig is None:
        continue
    H, W = orig.shape[:2]
    pred_px = pts * np.array([W - 1, H - 1])
    for x, y in pred_px:
        cv2.circle(orig, (int(x), int(y)), 4, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"real_{img_name}"), orig)

    spread = pts.std(axis=0).mean()
    spreads.append(spread)

if spreads:
    avg_spread = np.mean(spreads)
    print(f"  Tested {len(spreads)} real images")
    print(f"  Average landmark spread: {avg_spread:.4f}")
    check(
        "Real images: landmarks spread across ear (std > 0.01)",
        avg_spread > 0.01,
        f"Spread={avg_spread:.4f}",
    )

# Verify small bounding box applies
print("\n  --- Test 2: Small Bounding Box ---")
dummy_img = np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
small_bbox = (125, 125, 175, 175)
try:
    pts = run_inference(model_23, dummy_img, small_bbox)
    check("Model handles 50x50 bbox without crashing", pts.shape == (NUM_LANDMARKS, 2))
    check(
        "Small bbox: output coords in [0,1]",
        pts.min() >= 0.0 and pts.max() <= 1.0,
        f"Range: [{pts.min():.4f},{pts.max():.4f}]",
    )
except Exception as e:
    check("Model handles 50x50 bbox without crashing", False, str(e))


# Verify very dark image applies
print("\n  --- Test 3: Very Dark Image ---")
dark_img = np.zeros((400, 400, 3), dtype=np.uint8)
dark_bbox = (50, 50, 350, 350)
try:
    pts = run_inference(model_23, dark_img, dark_bbox)
    check(
        "Model handles all-black image without crashing",
        pts.shape == (NUM_LANDMARKS, 2),
    )
    check("Dark image: output coords in [0,1]", pts.min() >= 0.0 and pts.max() <= 1.0)
except Exception as e:
    check("Model handles all-black image without crashing", False, str(e))

# Verify very bright image applies
print("\n  --- Test 4: Very Bright Image ---")
bright_img = np.full((400, 400, 3), 255, dtype=np.uint8)
bright_bbox = (50, 50, 350, 350)
try:
    pts = run_inference(model_23, bright_img, bright_bbox)
    check(
        "Model handles all-white image without crashing",
        pts.shape == (NUM_LANDMARKS, 2),
    )
    check("Bright image: output coords in [0,1]", pts.min() >= 0.0 and pts.max() <= 1.0)
except Exception as e:
    check("Model handles all-white image without crashing", False, str(e))

# Verify wide aspect ratio bbox (2:1) applies
print("\n  --- Test 5: Wide Aspect Ratio Bbox (2:1) ---")
wide_img = np.random.randint(50, 200, (300, 600, 3), dtype=np.uint8)
wide_bbox = (50, 50, 550, 250)
try:
    pts = run_inference(model_23, wide_img, wide_bbox)
    check(
        "Model handles wide aspect ratio bbox without crashing",
        pts.shape == (NUM_LANDMARKS, 2),
    )
except Exception as e:
    check("Model handles wide aspect ratio bbox without crashing", False, str(e))

# Summary
print(f"\n======== Summary ========")
print(f"  PASSED: {PASSED}")
print(f"  FAILED: {FAILED}")
print(f"  Visualizations saved to {OUTPUT_DIR}/")
