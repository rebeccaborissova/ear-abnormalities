import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

MEASUREMENTS_CSV = "/home/UFAD/rborissova/senior_project/ear-abnormalities/measurements/measurements.csv"
LANDMARKS_CSV    = "/home/UFAD/rborissova/senior_project/ear-abnormalities/measurements/landmarks.csv"
LABELS_CSV       = "/home/UFAD/rborissova/senior_project/BabyEar4k/diagnosis_result.csv"

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
FEATURE_COLS  = MEASUREMENT_COLS + LANDMARK_COLS  # 7 + 46 = 53

N_MEAS = len(MEASUREMENT_COLS)   # 7
N_LM   = len(LANDMARK_COLS)      # 46

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Data loading ───────────────────────────────────────────────────────────────

def parse_label(label_str):
    """
    Parse a label string of the form 'a+b+c'.
    Returns (a, b, c) as ints, or None if the format is invalid.
    """
    try:
        parts = str(label_str).strip().split('+')
        if len(parts) != 3:
            return None
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, AttributeError):
        return None

def normalize_landmarks(points):
    """Center on centroid, scale by ear height (lm0 to lm19)."""
    pts = np.array(points, dtype=np.float32)  # (23, 2)
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
            else:
                print(f"  [landmarks] Skipping {img_name}: expected 23 points, got {len(coords)}")
    return landmarks_dict

def extract_side(img_name):
    """
    Extract the ear side character from the image name.
    Assumes the side char ('L' or 'R') is at index 5, but validates it.
    """
    if len(img_name) > 5 and img_name[5] in ('L', 'R'):
        return img_name[5]
    # Fallback: search for explicit side marker
    for candidate in ('_L_', '_R_', '/L/', '/R/'):
        if candidate in img_name:
            return candidate[1]
    return None

def load_dataset(measurements_csv, landmarks_csv, labels_csv):
    meas           = pd.read_csv(measurements_csv)
    labels         = pd.read_csv(labels_csv)
    landmarks_dict = load_landmarks(landmarks_csv)

    skipped = {'no_side': 0, 'no_label_match': 0, 'bad_label': 0, 'no_landmarks': 0}
    rows = []

    for _, meas_row in meas.iterrows():
        img_name = meas_row['image_path']

        # FIX: validate side extraction instead of blindly indexing
        side = extract_side(img_name)
        if side is None:
            skipped['no_side'] += 1
            continue

        # FIX: use exact equality check to avoid false-positive endswith matches
        col = f'dictory_{side}'
        if col not in labels.columns:
            skipped['no_label_match'] += 1
            continue
        label_row = labels[labels[col] == img_name]
        if label_row.empty:
            # Fallback to endswith only if exact match fails
            label_row = labels[labels[col].str.endswith('/' + img_name, na=False)]
        if label_row.empty:
            skipped['no_label_match'] += 1
            continue

        # FIX: safe label parsing with validation
        parsed = parse_label(label_row.iloc[0][f'{side}_merge'])
        if parsed is None:
            skipped['bad_label'] += 1
            continue
        a, b, c = parsed

        if img_name not in landmarks_dict:
            skipped['no_landmarks'] += 1
            continue

        # Clamp label_a to [0, 4] once, here
        a = min(a, 4)

        entry = {
            'image_path': img_name,
            'label_a': a,
            'label_b': b,
            'label_c': c,
        }
        for col in MEASUREMENT_COLS:
            entry[col] = meas_row[col]

        pts = landmarks_dict[img_name]
        for i in range(23):
            entry[f'lm_{i}_x'] = pts[i, 0]
            entry[f'lm_{i}_y'] = pts[i, 1]

        rows.append(entry)

    print(f"Skipped rows: {skipped}")
    return pd.DataFrame(rows)

# ── Dataset ────────────────────────────────────────────────────────────────────

class EarDataset(Dataset):
    def __init__(self, X, y_a):
        self.X   = torch.tensor(X, dtype=torch.float32)
        self.y_a = torch.tensor(y_a, dtype=torch.long)

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y_a[i]

# ── Model ──────────────────────────────────────────────────────────────────────

class EarNet(nn.Module):
    """
    Two-branch network:
      - Measurement branch: processes the 7 scalar measurements
      - Landmark branch:    processes the 46 (x,y) landmark coords
    Branches are fused, then classified.
    """
    def __init__(self, n_meas=N_MEAS, n_lm=N_LM, dropout=0.3):
        super().__init__()

        # FIX: separate branches for structurally different input types
        self.meas_branch = nn.Sequential(
            nn.Linear(n_meas, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.lm_branch = nn.Sequential(
            nn.Linear(n_lm, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # FIX: fusion head with dropout for regularization
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_a = nn.Linear(32, 5)  # classes: 0, 1, 2, 3, 4+

    def forward(self, x):
        x_meas = x[:, :N_MEAS]
        x_lm   = x[:, N_MEAS:]
        m = self.meas_branch(x_meas)
        l = self.lm_branch(x_lm)
        fused = self.fusion(torch.cat([m, l], dim=1))
        return self.head_a(fused)

# ── Train ──────────────────────────────────────────────────────────────────────

def train(df):
    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_a   = df['label_a'].values  # already clipped in load_dataset

    # FIX: three-way split — train / val (tuning) / test (final report)
    X_tr_val, X_test, a_tr_val, a_test = train_test_split(
        X_all, y_a, test_size=0.15, random_state=SEED, stratify=y_a
    )
    X_tr, X_val, a_tr, a_val = train_test_split(
        X_tr_val, a_tr_val, test_size=0.15 / 0.85, random_state=SEED, stratify=a_tr_val
    )

    # FIX: fit scaler ONLY on training data, then apply to val and test
    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X_tr)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"Split sizes — train: {len(X_tr)}, val: {len(X_val)}, test: {len(X_test)}")

    train_loader = DataLoader(EarDataset(X_tr, a_tr),   batch_size=32, shuffle=True)
    val_loader   = DataLoader(EarDataset(X_val, a_val), batch_size=32)
    test_loader  = DataLoader(EarDataset(X_test, a_test), batch_size=32)

    # Class-balanced loss weights
    classes      = np.unique(a_tr)
    weights      = compute_class_weight('balanced', classes=classes, y=a_tr)
    full_weights = np.ones(5, dtype=np.float32)
    for cls, w in zip(classes, weights):
        full_weights[cls] = w

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(full_weights))
    model     = EarNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # FIX: learning rate scheduler — reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    # FIX: save best checkpoint, not last epoch
    best_val_loss  = float('inf')
    best_state     = None
    patience_count = 0
    EARLY_STOP     = 40  # epochs without improvement before stopping

    for epoch in range(300):
        # ── training step ──
        model.train()
        train_loss = 0.0
        for X_b, la in train_loader:
            pa   = model(X_b)
            loss = criterion(pa, la)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── validation step ──
        model.eval()
        val_loss = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_b, la in val_loader:
                pa = model(X_b)
                val_loss += criterion(pa, la).item()
                all_preds.extend(pa.argmax(1).tolist())
                all_true.extend(la.tolist())
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # ── checkpoint best model ──
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if (epoch + 1) % 20 == 0:
            print(f"\nEpoch {epoch+1}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            print(classification_report(all_true, all_preds, zero_division=0))

        # FIX: early stopping
        if patience_count >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {EARLY_STOP} epochs)")
            break

    # ── final evaluation on held-out test set using best checkpoint ──
    print("\n=== Final evaluation on TEST SET (best checkpoint) ===")
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, la in test_loader:
            pa = model(X_b)
            all_preds.extend(pa.argmax(1).tolist())
            all_true.extend(la.tolist())
    print(classification_report(all_true, all_preds, zero_division=0))

    torch.save({
        'model_state_dict': best_state,
        'scaler': scaler,
    }, 'ear_model.pt')
    print("Best model saved to ear_model.pt")
    return model, scaler

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_dataset(MEASUREMENTS_CSV, LANDMARKS_CSV, LABELS_CSV)
    print(f"Loaded {len(df)} samples")
    print(df[['label_a', 'label_b', 'label_c']].value_counts())
    model, scaler = train(df)