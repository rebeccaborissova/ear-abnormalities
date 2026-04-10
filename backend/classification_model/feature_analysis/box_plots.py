import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    cfg = yaml.safe_load(f)

MEASUREMENTS_CSV = "../landmark_model/postprocess/measurements.csv"
LABELS_CSV = cfg['diagnosis_csv']

MEASUREMENT_COLS = [
    'dist_4_17_norm',
    'dist_20_21_norm',
    'dist_0_8_norm',
    'dist_21_22_norm',
    'curvature_angle_deg',
    'arc_length_norm',
    'chord_arc_ratio',
]

TARGET_COL = "label_a"
VALID_CLASSES = [0, 1, 2, 3, 4]


def parse_label(label_str):
    try:
        parts = str(label_str).strip().split('+')
        if len(parts) != 3:
            return None
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, AttributeError):
        return None


def extract_side(img_name):
    if len(img_name) > 5 and img_name[5] in ("L", "R"):
        return img_name[5]
    for candidate in ("_L_", "_R_", "/L/", "/R/"):
        if candidate in img_name:
            return candidate[1]
    return None


def load_dataset(measurements_csv, labels_csv):
    meas = pd.read_csv(measurements_csv)
    labels = pd.read_csv(labels_csv)

    skipped = {"no_side": 0, "no_label_match": 0, "bad_label": 0, "invalid_class": 0}
    rows = []

    for _, meas_row in meas.iterrows():
        img_name = meas_row["image_path"]

        side = extract_side(img_name)
        if side is None:
            skipped["no_side"] += 1
            continue

        col = f"dictory_{side}"
        if col not in labels.columns:
            skipped["no_label_match"] += 1
            continue

        label_row = labels[labels[col] == img_name]
        if label_row.empty:
            label_row = labels[labels[col].str.endswith("/" + img_name, na=False)]
        if label_row.empty:
            skipped["no_label_match"] += 1
            continue

        parsed = parse_label(label_row.iloc[0][f"{side}_merge"])
        if parsed is None:
            skipped["bad_label"] += 1
            continue

        a, b, c = parsed
        if a not in VALID_CLASSES:
            skipped["invalid_class"] += 1
            continue

        entry = {"image_path": img_name, TARGET_COL: a}
        for col_name in MEASUREMENT_COLS:
            entry[col_name] = meas_row[col_name]
        rows.append(entry)

    df = pd.DataFrame(rows)
    print(f"Skipped: {skipped}")
    return df


def plot_distributions(df):
    classes = sorted(df[TARGET_COL].unique())
    class_labels = [f"Class {c}\n(n={len(df[df[TARGET_COL]==c])})" for c in classes]
    palette = sns.color_palette("Set2", len(classes))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, col in enumerate(MEASUREMENT_COLS):
        sns.boxplot(
            data=df, x=TARGET_COL, y=col,
            palette=palette, ax=axes[i],
            order=classes, inner="box",
        )
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Class")
        axes[i].set_xticklabels(class_labels, fontsize=8)
    axes[-1].set_visible(False)
    plt.suptitle("Measurement Distributions by Class — Box Plots", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plot_boxes.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved plot_boxes.png")


if __name__ == "__main__":
    df = load_dataset(MEASUREMENTS_CSV, LABELS_CSV)
    plot_distributions(df)