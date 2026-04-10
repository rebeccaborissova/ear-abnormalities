import os
import torch
import cv2
import numpy as np
import pandas as pd
import torch.nn.functional as F

import yaml
from adult_model import get_model, soft_argmax_2d
from adult_utils import load_model

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

_inf  = cfg['infant_training']
_pred = cfg['predict_unlabeled']

MODEL_PATH = _inf['output_checkpoint']
NUM_LANDMARKS = _inf['num_landmarks']
NUM_STAGES = _inf['num_stages']
INPUT_SIZE = _inf['input_size']
IMAGES_DIR = cfg['images_dir']
OUTPUT_DIR = _pred['output_dir']
LANDMARKS_CSV = _pred['landmarks_csv']
CONFIDENCES_CSV = _pred['confidences_csv']

KEY_LANDMARKS = [3, 4, 5, 20, 21, 22]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = get_model(NUM_LANDMARKS, NUM_STAGES).to(device)
model = load_model(model, MODEL_PATH)
model.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)

img_paths = [
    os.path.join(IMAGES_DIR, f)
    for f in sorted(os.listdir(IMAGES_DIR))
    if f.lower().endswith(".jpg") or f.lower().endswith(".png")
]
print(f"Found {len(img_paths)} images")

landmarks_rows   = []
confidences_rows = []

for idx, img_path in enumerate(img_paths):
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)

    if img is None:
        print(f"ERROR: Could not read {img_path}, skipping")
        continue

    h, w = img.shape[:2]

    # Preprocess
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_resized = img_resized / 255.0
    img_chw     = np.transpose(img_resized, (2, 0, 1))
    img_tensor  = torch.tensor(img_chw, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        stage_outputs = model(img_tensor)
        # stage_outputs: (1, 6, 23, H, W)

        # Get predicted coordinates for ALL stages
        all_stage_coords = []
        for s in range(stage_outputs.shape[1]):
            heatmap_s = stage_outputs[:, s]  # (1, 23, H, W)
            coords_s  = soft_argmax_2d(heatmap_s, normalize=True)  # (1, 23, 2)
            all_stage_coords.append(coords_s.cpu().numpy().squeeze())  # (23, 2)

        all_stage_coords = np.array(all_stage_coords)  # (6, 23, 2)

        # Use only later stages for stability check — early stages are always noisy
        late_stages = all_stage_coords[2:]  # stages 3-6, shape (4, 23, 2)

        # Std across late stages per landmark — low std = stable = confident
        stage_std        = late_stages.std(axis=0)              # (23, 2)
        stage_std_per_lm = stage_std.mean(axis=1)               # (23,) — one value per landmark
        confidences      = 1.0 / (1.0 + stage_std_per_lm)      # (23,) — high = stable = confident

        # Final prediction from last stage
        final_heatmaps = stage_outputs[:, -1]
        pred = soft_argmax_2d(final_heatmaps, normalize=True).cpu().numpy().squeeze()

    # Denormalize to original image size
    pred_px = pred * np.array([w - 1, h - 1])

    # --- Landmarks row ---
    lm_parts = [img_name]
    for (x, y) in pred_px:
        lm_parts.append(f"{x:.4f} {y:.4f}")
    landmarks_rows.append(", ".join(lm_parts))

    # --- Confidences row ---
    conf_row = {"image_path": img_name}
    for i, conf in enumerate(confidences):
        conf_row[f"conf_lm_{i}"] = float(conf)
    conf_row["conf_mean"]     = float(confidences.mean())
    conf_row["conf_min"]      = float(confidences.min())
    conf_row["conf_key_mean"] = float(confidences[KEY_LANDMARKS].mean())
    conf_row["conf_key_min"]  = float(confidences[KEY_LANDMARKS].min())
    confidences_rows.append(conf_row)

    # Visualize — green = confident, red = uncertain
    img_vis = img.copy()
    for i, (x, y) in enumerate(pred_px):
        color = (0, 255, 0) if confidences[i] > 0.5 else (0, 0, 255)
        cv2.circle(img_vis, (int(x), int(y)), 3, color, -1)
        cv2.putText(img_vis, str(i), (int(x)+4, int(y)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, img_vis)

    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(img_paths)}")

# Save landmarks CSV
with open(LANDMARKS_CSV, "w") as f:
    for row in landmarks_rows:
        f.write(row + "\n")
print(f"\nSaved {len(landmarks_rows)} landmarks to {LANDMARKS_CSV}")

# Save confidences CSV
conf_df = pd.DataFrame(confidences_rows)
conf_df.to_csv(CONFIDENCES_CSV, index=False)
print(f"Saved confidences to {CONFIDENCES_CSV}")

# Print confidence distribution to help pick threshold
print(f"\n=== Confidence Distribution ===")
print(f"  Mean confidence (all landmarks):  {conf_df['conf_mean'].mean():.4f}")
print(f"  Mean confidence (key landmarks):  {conf_df['conf_key_mean'].mean():.4f}")
print(f"  Min confidence (key landmarks):   {conf_df['conf_key_min'].mean():.4f}")
print(f"\n  % images with key conf_min > 0.1: {(conf_df['conf_key_min'] > 0.1).mean():.1%}")
print(f"  % images with key conf_min > 0.2: {(conf_df['conf_key_min'] > 0.2).mean():.1%}")
print(f"  % images with key conf_min > 0.3: {(conf_df['conf_key_min'] > 0.3).mean():.1%}")
print(f"  % images with key conf_min > 0.5: {(conf_df['conf_key_min'] > 0.5).mean():.1%}")

print(f"\n  Key landmark confidence percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"    {p}th percentile: {np.percentile(conf_df['conf_key_min'], p):.4f}")