"""
ResNet-18 ear severity classifier.
Predicts label_a in {0, 1, 2, 3, 4} directly from raw ear images.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, balanced_accuracy_score
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────

IMAGES_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/images"
LABELS_CSV = "/home/UFAD/rborissova/senior_project/BabyEar4k/diagnosis_result.csv"

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_EPOCHS  = 60
LR          = 1e-4
WEIGHT_DECAY= 1e-4
DROPOUT     = 0.5
SEED        = 42
CKPT_PATH   = "severity_resnet18_best.pth"
EARLY_STOP  = 15   # epochs without val improvement

torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Label loading ──────────────────────────────────────────────────────────────

def parse_label(label_str):
    try:
        parts = str(label_str).strip().split('+')
        if len(parts) != 3:
            return None
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, AttributeError):
        return None

def extract_side(img_name):
    if len(img_name) > 5 and img_name[5] in ('L', 'R'):
        return img_name[5]
    for candidate in ('_L_', '_R_', '/L/', '/R/'):
        if candidate in img_name:
            return candidate[1]
    return None

def load_labels(labels_csv, images_dir):
    """
    Returns a list of (image_filename, label_a) for every image
    that exists on disk and has a valid label.
    """
    labels = pd.read_csv(labels_csv)
    rows   = []
    skipped = 0

    for _, row in labels.iterrows():
        for side in ('L', 'R'):
            col = f'dictory_{side}'
            if col not in labels.columns:
                continue
            img_name = str(row[col]).strip()
            if not img_name or img_name == 'nan':
                continue

            # Check image exists
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                # Try just the basename
                img_path = os.path.join(images_dir, os.path.basename(img_name))
                if not os.path.exists(img_path):
                    skipped += 1
                    continue
                img_name = os.path.basename(img_name)

            parsed = parse_label(row.get(f'{side}_merge', None))
            if parsed is None:
                skipped += 1
                continue
            a, _, _ = parsed
            a = min(a, 4)
            rows.append((img_name, a))

    print(f"Loaded {len(rows)} image-label pairs  ({skipped} skipped)")

    df = pd.DataFrame(rows, columns=['image', 'label_a'])
    print("\nClass distribution:")
    print(df['label_a'].value_counts().sort_index())
    return df

# ── Dataset ────────────────────────────────────────────────────────────────────

# ImageNet normalization — correct for pretrained ResNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class EarImageDataset(Dataset):
    def __init__(self, df, images_dir, transform):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image'])
        img      = Image.open(img_path).convert('RGB')
        img      = self.transform(img)
        label    = torch.tensor(row['label_a'], dtype=torch.long)
        return img, label

# ── Model ──────────────────────────────────────────────────────────────────────

def get_model(num_classes=5, dropout=DROPOUT):
    """
    ResNet-18 pretrained on ImageNet, with a custom classification head.
    Only the last ResNet block + head are unfrozen initially.
    """
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all layers
    for param in m.parameters():
        param.requires_grad = False

    # Unfreeze layer4 (last residual block) and the final fc
    for param in m.layer4.parameters():
        param.requires_grad = True

    # Replace classifier head
    in_features = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return m

# ── Training ───────────────────────────────────────────────────────────────────

def make_weights(labels_train):
    """Square-root dampened class weights — less aggressive than 'balanced'."""
    counts = np.bincount(labels_train, minlength=5).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    w = 1.0 / np.sqrt(counts)
    w = w / w.sum() * 5   # normalize so weights sum to n_classes
    return torch.tensor(w, dtype=torch.float32)

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, all_preds, all_true = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    return total_loss / len(loader), all_true, all_preds

def train(df):
    # Three-way split: 70 / 15 / 15
    df_trainval, df_test = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df['label_a']
    )
    df_train, df_val = train_test_split(
        df_trainval, test_size=0.15/0.85, random_state=SEED, stratify=df_trainval['label_a']
    )
    print(f"\nSplit — train: {len(df_train)}  val: {len(df_val)}  test: {len(df_test)}")

    train_ds = EarImageDataset(df_train, IMAGES_DIR, train_transforms)
    val_ds   = EarImageDataset(df_val,   IMAGES_DIR, val_transforms)
    test_ds  = EarImageDataset(df_test,  IMAGES_DIR, val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model     = get_model().to(device)
    weights   = make_weights(df_train['label_a'].values).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    best_val_loss  = float('inf')
    best_state     = None
    patience_count = 0

    for epoch in range(NUM_EPOCHS):
        tr_loss, _, _           = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, vt, vp        = run_epoch(model, val_loader,   criterion)
        val_bal_acc             = balanced_accuracy_score(vt, vp)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"val_bal_acc={val_bal_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
            torch.save({'model_state_dict': best_state, 'best_val_loss': best_val_loss}, CKPT_PATH)
            print(f"  --> Saved checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # ── Final evaluation on held-out test set ──────────────────────────────────
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION (best checkpoint)")
    print("="*60)
    model.load_state_dict(best_state)
    _, tt, tp = run_epoch(model, test_loader, criterion)
    print(classification_report(tt, tp, zero_division=0))
    print(f"Balanced accuracy: {balanced_accuracy_score(tt, tp):.3f}")

    return model

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df    = load_labels(LABELS_CSV, IMAGES_DIR)
    model = train(df)