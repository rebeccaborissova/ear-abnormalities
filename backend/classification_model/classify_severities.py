import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from imblearn.over_sampling import SMOTE
from utils.classification_utils import load_config, parse_label, extract_side

cfg = load_config()
LABELS_CSV = cfg['diagnosis_csv']
MEASUREMENTS_CSV = "../landmark_model/postprocess/measurements.csv"
LANDMARKS_CSV = "../landmark_model/postprocess/landmarks.csv"

MEASUREMENT_COLS = [
    'dist_4_17_norm',
    'dist_20_21_norm',
    'dist_0_8_norm',
    'dist_21_22_norm',
    'curvature_angle_deg',
    'arc_length_norm',
    'chord_arc_ratio',
]

LANDMARK_COLS = [f'lm_{i}_{axis}' for i in range(23) for axis in ['x', 'y']]

KEY_LANDMARKS = [3, 4, 5, 20, 21, 22]

KEY_RELATIVE_COLS = [
    f'key_rel_{axis}_{i}_{j}'
    for i, j in combinations(KEY_LANDMARKS, 2)
    for axis in ['x', 'y']
]

# 20 angle features from all unique triplets of the 6 key landmarks
ANGLE_TRIPLETS = list(combinations(KEY_LANDMARKS, 3))
ANGLE_COLS = [f'angle_deg_{i}_{j}_{k}' for i, j, k in ANGLE_TRIPLETS]

COMBINED_COLS = MEASUREMENT_COLS + KEY_RELATIVE_COLS
COMBINED_WITH_ANGLES_COLS = MEASUREMENT_COLS + KEY_RELATIVE_COLS + ANGLE_COLS

SEED = 42
TARGET_COL = "label_a"
VALID_CLASSES = {0, 1, 2}
N_FOLDS = 5


def normalize_landmarks(points):
    pts = np.array(points, dtype=np.float32)
    centroid = pts.mean(axis=0)
    pts -= centroid
    ear_height = np.linalg.norm(pts[0] - pts[19])
    if ear_height > 0:
        pts /= ear_height
    return pts


def compute_angle_deg(a, b, c):
    """
    Returns angle ABC in degrees, where the angle is at point B.
    """
    ba = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    bc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cos_theta = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def load_landmarks(landmarks_csv):
    landmarks_dict = {}
    with open(landmarks_csv, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            img_name = parts[0].strip()
            coords = []
            for part in parts[1:]:
                xy = part.strip().split()
                if len(xy) == 2:
                    try:
                        coords.append((float(xy[0]), float(xy[1])))
                    except ValueError:
                        continue
            if len(coords) == 23:
                landmarks_dict[img_name] = normalize_landmarks(coords)
            else:
                print(f"[landmarks] Skipping {img_name}: expected 23 points, got {len(coords)}")
    return landmarks_dict


def load_dataset(measurements_csv, landmarks_csv, labels_csv):
    meas = pd.read_csv(measurements_csv)
    labels = pd.read_csv(labels_csv)
    landmarks_dict = load_landmarks(landmarks_csv)

    skipped = {
        "no_side": 0,
        "no_label_match": 0,
        "bad_label": 0,
        "no_landmarks": 0,
        "invalid_target_class": 0,
    }
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

        if img_name not in landmarks_dict:
            skipped["no_landmarks"] += 1
            continue

        entry = {
            "image_path": img_name,
            "label_a": a,
            "label_b": b,
            "label_c": c,
        }

        target_value = entry[TARGET_COL]
        if target_value not in VALID_CLASSES:
            skipped["invalid_target_class"] += 1
            continue

        for col_name in MEASUREMENT_COLS:
            entry[col_name] = meas_row[col_name]

        pts = landmarks_dict[img_name]
        for i in range(23):
            entry[f"lm_{i}_x"] = pts[i, 0]
            entry[f"lm_{i}_y"] = pts[i, 1]

        rows.append(entry)

    df = pd.DataFrame(rows)
    print(f"Skipped rows: {skipped}")
    return df


def add_key_relative_positions(df):
    new_cols = {}
    for i, j in combinations(KEY_LANDMARKS, 2):
        new_cols[f'key_rel_x_{i}_{j}'] = df[f'lm_{i}_x'] - df[f'lm_{j}_x']
        new_cols[f'key_rel_y_{i}_{j}'] = df[f'lm_{i}_y'] - df[f'lm_{j}_y']
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_angle_features(df, angle_triplets=ANGLE_TRIPLETS):
    new_cols = {}

    for i, j, k in angle_triplets:
        values = []
        for _, row in df.iterrows():
            a = (row[f'lm_{i}_x'], row[f'lm_{i}_y'])
            b = (row[f'lm_{j}_x'], row[f'lm_{j}_y'])
            c = (row[f'lm_{k}_x'], row[f'lm_{k}_y'])
            values.append(compute_angle_deg(a, b, c))

        new_cols[f'angle_deg_{i}_{j}_{k}'] = values

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def make_rf():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=1,
    )


def _run_cv_loop(X, y, model, n_folds, seed):
    classes = sorted(y.unique())
    n_classes = len(classes)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    smote = SMOTE(random_state=seed)

    all_preds = np.zeros(len(y), dtype=int)
    all_probas = np.zeros((len(y), n_classes))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_val_imp = imputer.transform(X_val)

        X_tr_res, y_tr_res = smote.fit_resample(X_tr_imp, y_tr)

        m = clone(model)
        m.fit(X_tr_res, y_tr_res)
        all_preds[val_idx] = m.predict(X_val_imp)
        all_probas[val_idx] = m.predict_proba(X_val_imp)
        print(f"  Fold {fold+1} done")

    return all_preds, all_probas, classes, n_classes


def _print_cv_results(y, all_preds, all_probas, classes, n_classes, label):
    print(f"\n=== Classification Report — {label} ===")
    print(classification_report(y, all_preds, labels=classes, zero_division=0))
    print(f"=== Confusion Matrix — {label} (rows=true, cols=pred) ===")
    print(confusion_matrix(y, all_preds, labels=classes))

    if n_classes == 2:
        print(f"ROC-AUC (binary): {roc_auc_score(y, all_probas[:, 1]):.3f}")
    else:
        print(f"ROC-AUC (macro, one-vs-rest): {roc_auc_score(y, all_probas, multi_class='ovr', average='macro'):.3f}")
        print(f"ROC-AUC (macro, one-vs-one):  {roc_auc_score(y, all_probas, multi_class='ovo', average='macro'):.3f}")


def run_cv(X, y, label="", model=None):
    if model is None:
        model = make_rf()
    print(f"\nRunning {N_FOLDS}-fold CV — {label} ({X.shape[1]} features)...")
    all_preds, all_probas, classes, n_classes = _run_cv_loop(X, y, model, N_FOLDS, SEED)
    _print_cv_results(y, all_preds, all_probas, classes, n_classes, label)
    return all_preds, all_probas


def run_decision_tree(X, y):
    print("\n" + "=" * 50)
    print("=== Decision Tree — Interpretable Rules ===")
    print("=" * 50)

    tree = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=SEED)
    all_preds, all_probas, classes, n_classes = _run_cv_loop(X, y, tree, N_FOLDS, SEED)
    _print_cv_results(y, all_preds, all_probas, classes, n_classes, "Decision Tree")

    smote = SMOTE(random_state=SEED)
    imputer_full = SimpleImputer(strategy="median")
    X_imp_full = imputer_full.fit_transform(X)
    X_res_full, y_res_full = smote.fit_resample(X_imp_full, y)
    tree.fit(X_res_full, y_res_full)

    print("\n=== Decision Tree Rules (fitted on full dataset) ===")
    print(export_text(tree, feature_names=list(X.columns)))

    return all_preds, all_probas


if __name__ == "__main__":
    df = load_dataset(MEASUREMENTS_CSV, LANDMARKS_CSV, LABELS_CSV)

    print(f"Loaded {len(df)} samples")
    print("\n=== Class Counts ===")
    print(df[TARGET_COL].value_counts().sort_index())

    df = add_key_relative_positions(df)
    df = add_angle_features(df)

    y = df[TARGET_COL].astype(int).copy()

    X_meas = df[MEASUREMENT_COLS].copy()
    run_cv(X_meas, y, label="Measurements only (7 features)")

    X_key = df[KEY_RELATIVE_COLS].copy()
    run_cv(X_key, y, label=f"Key landmarks only ({len(KEY_RELATIVE_COLS)} features)")

    X_combined = df[COMBINED_COLS].copy()
    run_cv(X_combined, y, label=f"Measurements + key landmarks ({len(COMBINED_COLS)} features)")

    X_combined_angles = df[COMBINED_WITH_ANGLES_COLS].copy()
    run_cv(
        X_combined_angles,
        y,
        label=f"Measurements + key landmarks + angles ({len(COMBINED_WITH_ANGLES_COLS)} features)"
    )

    run_decision_tree(X_combined_angles, y)