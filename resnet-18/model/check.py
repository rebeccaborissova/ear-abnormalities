import numpy as np
import pandas as pd
from collections import Counter

MEASUREMENTS_CSV = "/home/UFAD/rborissova/senior_project/ear-abnormalities/measurements/measurements.csv"
LANDMARKS_CSV    = "/home/UFAD/rborissova/senior_project/ear-abnormalities/measurements/landmarks.csv"
LABELS_CSV       = "/home/UFAD/rborissova/senior_project/BabyEar4k/diagnosis_result.csv"

MEASUREMENT_COLS = [
    'dist_4_17_norm', 'dist_20_21_norm', 'dist_0_8_norm', 'dist_21_22_norm',
    'curvature_angle_deg', 'arc_length_norm', 'chord_arc_ratio',
]

# Set this to True if you want to collapse all values >4 into class 4
CLIP_TO_0_4 = True


def parse_label(label_str):
    try:
        parts = str(label_str).strip().split('+')
        if len(parts) != 3:
            return None
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, AttributeError):
        return None


def normalize_landmarks(points):
    pts = np.array(points, dtype=np.float32)
    centroid = pts.mean(axis=0)
    pts -= centroid
    ear_height = np.linalg.norm(pts[0] - pts[19])
    if ear_height > 0:
        pts /= ear_height
    return pts


def load_landmarks(landmarks_csv):
    landmarks_dict = {}
    with open(landmarks_csv, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            if not parts:
                continue
            img_name = parts[0].strip()
            coords = []
            for part in parts[1:]:
                xy = part.strip().split()
                if len(xy) == 2:
                    try:
                        coords.append((float(xy[0]), float(xy[1])))
                    except ValueError:
                        pass
            if len(coords) == 23:
                landmarks_dict[img_name] = normalize_landmarks(coords)
    return landmarks_dict


def extract_side(img_name):
    if len(img_name) > 5 and img_name[5] in ("L", "R"):
        return img_name[5]
    for candidate in ("_L_", "_R_", "/L/", "/R/"):
        if candidate in img_name:
            return candidate[1]
    return None


def load_dataset(measurements_csv, landmarks_csv, labels_csv):
    meas = pd.read_csv(measurements_csv)
    labels = pd.read_csv(labels_csv)
    landmarks_dict = load_landmarks(landmarks_csv)

    rows = []
    skipped = Counter()

    for _, meas_row in meas.iterrows():
        img_name = meas_row["image_path"]
        side = extract_side(img_name)
        if side is None:
            skipped["no_side"] += 1
            continue

        # Change this if "dictory_L/R" is a typo in your CSV
        col = f"dictory_{side}"
        if col not in labels.columns:
            skipped["missing_label_column"] += 1
            continue

        label_row = labels[labels[col] == img_name]
        if label_row.empty:
            label_row = labels[labels[col].str.endswith("/" + img_name, na=False)]
        if label_row.empty:
            skipped["no_label_match"] += 1
            continue

        merged_col = f"{side}_merge"
        if merged_col not in labels.columns:
            skipped["missing_merge_column"] += 1
            continue

        parsed = parse_label(label_row.iloc[0][merged_col])
        if parsed is None:
            skipped["bad_label_parse"] += 1
            continue

        a, b, c = parsed

        if CLIP_TO_0_4:
            a = min(a, 4)

        if img_name not in landmarks_dict:
            skipped["missing_landmarks"] += 1
            continue

        row = {
            "image_path": img_name,
            "label_a": a,
            "label_b": b,
            "label_c": c,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, skipped


def print_counts(df, col):
    counts = df[col].value_counts().sort_index()
    total = len(df)

    print(f"\nCounts for {col}:")
    for cls, n in counts.items():
        pct = 100.0 * n / total if total > 0 else 0.0
        print(f"  class {cls}: {n:4d} ({pct:5.1f}%)")

    print(f"  total   : {total}")


if __name__ == "__main__":
    df, skipped = load_dataset(MEASUREMENTS_CSV, LANDMARKS_CSV, LABELS_CSV)

    print(f"Loaded {len(df)} matched samples.")

    print_counts(df, "label_a")
    print_counts(df, "label_b")
    print_counts(df, "label_c")

    print("\nSkipped rows:")
    for reason, n in skipped.items():
        print(f"  {reason}: {n}")