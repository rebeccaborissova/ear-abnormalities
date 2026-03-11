"""
Step 1 diagnostic: does the feature set actually contain enough signal
to predict ordinal severity label_a?

TRUE ORDINAL VERSION:
- trains K-1 cumulative threshold models for classes 0..K-1
- for 5 classes (0..4), learns:
    P(y > 0), P(y > 1), P(y > 2), P(y > 3)
- decodes predictions in an ordinal-consistent way

Primary metrics:
- MAE
- Within-1 accuracy
- Quadratic weighted kappa
- Exact accuracy
- Balanced accuracy (after ordinal decoding)

Also keeps:
- binary check: normal (0) vs any deformity (1+)
- coarse 3-class check: normal / mild / severe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    cohen_kappa_score,
)
from sklearn.decomposition import PCA

# ── Copy these from your main script ──────────────────────────────────────────
MEASUREMENTS_CSV = "/home/UFAD/rborissova/senior_project/ear-abnormalities/measurements/measurements.csv"
LANDMARKS_CSV    = "/home/UFAD/rborissova/senior_project/ear-abnormalities/measurements/landmarks.csv"
LABELS_CSV       = "/home/UFAD/rborissova/senior_project/BabyEar4k/diagnosis_result.csv"

MEASUREMENT_COLS = [
    'dist_4_17_norm', 'dist_20_21_norm', 'dist_0_8_norm', 'dist_21_22_norm',
    'curvature_angle_deg', 'arc_length_norm', 'chord_arc_ratio',
]
LANDMARK_COLS = [f'lm_{i}_{axis}' for i in range(23) for axis in ['x', 'y']]
FEATURE_COLS  = MEASUREMENT_COLS + LANDMARK_COLS

SEED = 42
MIN_CLASS = 0
MAX_CLASS = 4
NUM_CLASSES = MAX_CLASS - MIN_CLASS + 1


# ── Data loading helpers ──────────────────────────────────────────────────────

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
    with open(landmarks_csv, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
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
    return landmarks_dict

def extract_side(img_name):
    if len(img_name) > 5 and img_name[5] in ('L', 'R'):
        return img_name[5]
    for candidate in ('_L_', '_R_', '/L/', '/R/'):
        if candidate in img_name:
            return candidate[1]
    return None

def load_dataset(measurements_csv, landmarks_csv, labels_csv):
    meas           = pd.read_csv(measurements_csv)
    labels         = pd.read_csv(labels_csv)
    landmarks_dict = load_landmarks(landmarks_csv)
    rows = []

    for _, meas_row in meas.iterrows():
        img_name = meas_row['image_path']
        side = extract_side(img_name)
        if side is None:
            continue

        # If this is a typo in your CSV, replace with the real column name.
        col = f'dictory_{side}'
        if col not in labels.columns:
            continue

        label_row = labels[labels[col] == img_name]
        if label_row.empty:
            label_row = labels[labels[col].str.endswith('/' + img_name, na=False)]
        if label_row.empty:
            continue

        parsed = parse_label(label_row.iloc[0][f'{side}_merge'])
        if parsed is None:
            continue

        a, b, c = parsed
        if img_name not in landmarks_dict:
            continue

        # Keep only if you're truly using classes 0..4.
        # If your true range is 0..7, update MAX_CLASS and remove/change this.
        a = min(a, 4)

        entry = {
            'image_path': img_name,
            'label_a': a,
            'label_b': b,
            'label_c': c,
        }

        for col_name in MEASUREMENT_COLS:
            entry[col_name] = meas_row[col_name]

        pts = landmarks_dict[img_name]
        for i in range(23):
            entry[f'lm_{i}_x'] = pts[i, 0]
            entry[f'lm_{i}_y'] = pts[i, 1]

        rows.append(entry)

    return pd.DataFrame(rows)


# ── True ordinal helpers ──────────────────────────────────────────────────────

def make_ordinal_targets(y, min_class=MIN_CLASS, max_class=MAX_CLASS):
    """
    For y in {0,1,2,3,4}, return targets for thresholds:
      t=0: y > 0
      t=1: y > 1
      t=2: y > 2
      t=3: y > 3

    Output shape: (N, K-1)
    """
    y = np.asarray(y).astype(int)
    thresholds = list(range(min_class, max_class))
    Y = np.stack([(y > t).astype(int) for t in thresholds], axis=1)
    return Y

def decode_ordinal_probs_monotone(probs, threshold=0.5):
    """
    probs shape: (N, K-1), where column j estimates P(y > j)

    Enforce ordinal consistency by counting how many thresholds are passed
    in order until the first failure.
    """
    probs = np.asarray(probs)
    passed = probs >= threshold
    preds = np.zeros(probs.shape[0], dtype=int)

    for i in range(probs.shape[0]):
        k = 0
        for j in range(probs.shape[1]):
            if passed[i, j]:
                k += 1
            else:
                break
        preds[i] = k

    return preds

def summarize_ordinal_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    within1 = np.mean(np.abs(y_pred - y_true) <= 1)
    exact_acc = np.mean(y_pred == y_true)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    return {
        "mae": mae,
        "within1": within1,
        "exact_acc": exact_acc,
        "bal_acc": bal_acc,
        "qwk": qwk,
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────

def run_true_ordinal_random_forest(X, y, feature_names):
    print("\n" + "=" * 60)
    print("1. TRUE ORDINAL RANDOM FOREST (cumulative thresholds)")
    print("=" * 60)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    Y_tr_ord = make_ordinal_targets(y_tr)
    Y_val_ord = make_ordinal_targets(y_val)

    threshold_models = []
    threshold_probs = []

    print("\nTraining threshold models:")
    for j, t in enumerate(range(MIN_CLASS, MAX_CLASS)):
        y_tr_bin = Y_tr_ord[:, j]
        y_val_bin = Y_val_ord[:, j]

        positives = int(y_tr_bin.sum())
        negatives = int((1 - y_tr_bin).sum())

        print(f"  Threshold {j}: predict (y > {t}) | positives={positives}, negatives={negatives}")

        clf = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=SEED + j,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr_bin)
        threshold_models.append(clf)

        probs_j = clf.predict_proba(X_val)[:, 1]
        threshold_probs.append(probs_j)

        pred_bin = (probs_j >= 0.5).astype(int)
        bal_acc_j = balanced_accuracy_score(y_val_bin, pred_bin)
        print(f"    balanced acc for (y > {t}): {bal_acc_j:.3f}")

    threshold_probs = np.stack(threshold_probs, axis=1)   # (N, K-1)
    y_pred = decode_ordinal_probs_monotone(threshold_probs, threshold=0.5)

    metrics = summarize_ordinal_metrics(y_val, y_pred)

    print("\nOrdinal metrics after monotone decoding:")
    print(f"MAE:               {metrics['mae']:.3f}")
    print(f"Within-1 accuracy: {metrics['within1']:.3f}")
    print(f"Exact accuracy:    {metrics['exact_acc']:.3f}")
    print(f"Balanced accuracy: {metrics['bal_acc']:.3f}")
    print(f"Quadratic kappa:   {metrics['qwk']:.3f}")

    print("\nClassification report (decoded ordinal predictions):")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax)
    ax.set_title(
        f"True Ordinal RF — MAE={metrics['mae']:.3f}, "
        f"within-1={metrics['within1']:.3f}, QWK={metrics['qwk']:.3f}"
    )
    plt.tight_layout()
    plt.savefig("confusion_true_ordinal_rf.png", dpi=150)
    print("Saved: confusion_true_ordinal_rf.png")

    # Error histogram
    err = y_pred - y_val
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(err.min() - 0.5, err.max() + 1.5, 1)
    ax.hist(err, bins=bins, edgecolor='black')
    ax.set_xlabel("Ordinal prediction error (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title("True ordinal prediction errors")
    plt.tight_layout()
    plt.savefig("true_ordinal_error_hist.png", dpi=150)
    print("Saved: true_ordinal_error_hist.png")

    # Feature importances averaged across threshold models
    importances = np.mean(
        np.stack([m.feature_importances_ for m in threshold_models], axis=0),
        axis=0
    )
    imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])

    print("\nTop 15 most important features (mean over thresholds):")
    for name, score in imp[:15]:
        print(f"  {score:.4f}  {name}")

    # Save raw threshold probability diagnostics
    print("\nMean predicted probabilities for each threshold:")
    for j, t in enumerate(range(MIN_CLASS, MAX_CLASS)):
        print(f"  P(y > {t}) mean = {threshold_probs[:, j].mean():.3f}")

    return metrics, threshold_models

def run_binary_check(X, y):
    print("\n" + "=" * 60)
    print("2. BINARY CHECK: normal (0) vs any deformity (1+)")
    print("=" * 60)

    y_bin = (y > 0).astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_bin, test_size=0.2, random_state=SEED, stratify=y_bin
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_val)

    bal_acc = balanced_accuracy_score(y_val, preds)
    print(f"\nBalanced accuracy: {bal_acc:.3f}  (random chance = 0.500)")
    print(classification_report(y_val, preds, zero_division=0))
    return bal_acc

def run_coarse_check(X, y):
    print("\n" + "=" * 60)
    print("3. COARSE 3-CLASS: normal / mild (1-2) / severe (3-4)")
    print("=" * 60)

    y_coarse = np.where(y == 0, 0, np.where(y <= 2, 1, 2))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_coarse, test_size=0.2, random_state=SEED, stratify=y_coarse
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_val)

    bal_acc = balanced_accuracy_score(y_val, preds)
    print(f"\nBalanced accuracy: {bal_acc:.3f}  (random chance = 0.333)")
    print(classification_report(y_val, preds, zero_division=0))
    return bal_acc

def run_pca_plot(X, y):
    print("\n" + "=" * 60)
    print("4. PCA PLOT — visual class separability")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['steelblue', 'orange', 'green', 'red', 'purple']

    unique_classes = sorted(np.unique(y))
    for cls in unique_classes:
        mask = y == cls
        color = colors[cls] if cls < len(colors) else None
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=f'class {cls} (n={mask.sum()})',
            alpha=0.4,
            s=15,
            c=color,
        )

    ax.set_xlabel(f'PC1 ({var[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({var[1]:.1%} var)')
    ax.set_title('PCA of features — colored by ordinal label_a')
    ax.legend()
    plt.tight_layout()
    plt.savefig("pca_features.png", dpi=150)
    print(f"Saved: pca_features.png  (PC1+PC2 explain {sum(var):.1%} of variance)")
    print("Heavy overlap usually means nearby severity levels are not well separated by these features.")

def print_interpretation(ord_metrics, rf_binary, rf_coarse):
    mae = ord_metrics["mae"]
    within1 = ord_metrics["within1"]
    exact_acc = ord_metrics["exact_acc"]
    bal_acc = ord_metrics["bal_acc"]
    qwk = ord_metrics["qwk"]

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)

    print(f"\n  Ordinal MAE            : {mae:.3f}")
    print(f"  Within-1 accuracy      : {within1:.3f}")
    print(f"  Exact accuracy         : {exact_acc:.3f}")
    print(f"  Balanced accuracy      : {bal_acc:.3f}")
    print(f"  Quadratic weighted κ   : {qwk:.3f}")
    print(f"  Binary balanced acc    : {rf_binary:.3f}")
    print(f"  Coarse 3-class bal acc : {rf_coarse:.3f}")
    print()

    if rf_binary < 0.65:
        print("⛔ VERDICT: Features have very weak signal even for binary separation.")
        print("   → Your landmarks/measurements cannot reliably distinguish normal from abnormal ears.")
        print("   → Image features are likely needed.")
        return

    if within1 < 0.65 or qwk < 0.25:
        print("⚠️  VERDICT: Features have weak ordinal signal.")
        print("   → Even with true ordinal modeling, the model struggles.")
        print("   → This suggests the feature representation is too weak or too noisy for severity grading.")
        print("   → Next check: compare ground-truth landmarks vs predicted landmarks on the same subset.")
        return

    if within1 >= 0.65 and within1 < 0.85:
        print("✅ VERDICT: Features contain moderate ordinal signal.")
        print("   → The model often lands near the correct severity, even if exact class is hard.")
        print("   → Next steps:")
        print("     1. Try stronger tabular models per threshold (XGBoost/CatBoost if available)")
        print("     2. Try an ordinal neural net")
        print("     3. Add image features if exact grading is still poor")
        return

    print("✅ VERDICT: Features contain strong ordinal signal.")
    print("   → The feature set is informative for ordered severity prediction.")
    print("   → If later performance is still disappointing, model choice/training is likely the bottleneck.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset(MEASUREMENTS_CSV, LANDMARKS_CSV, LABELS_CSV)
    print(f"Loaded {len(df)} samples\n")

    if len(df) == 0:
        raise RuntimeError(
            "No samples loaded. Check CSV paths, label matching, and whether "
            "'dictory_L' / 'dictory_R' are the correct column names."
        )

    X = StandardScaler().fit_transform(df[FEATURE_COLS].values)
    y = df['label_a'].values.astype(int)

    ord_metrics, ordinal_models = run_true_ordinal_random_forest(X, y, FEATURE_COLS)
    bal_binary = run_binary_check(X, y)
    bal_coarse = run_coarse_check(X, y)
    run_pca_plot(X, y)
    print_interpretation(ord_metrics, bal_binary, bal_coarse)