import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

LANDMARKS_TRAIN_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/landmarks/train"
LANDMARKS_TEST_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/landmarks/test"
IMAGES_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/images"

LANDMARK_MAPPING = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 37,
    21: 25,
}

def load_all_annotations(num_landmarks):
    all_annotations = {}

    for landmarks_dir in [LANDMARKS_TRAIN_DIR, LANDMARKS_TEST_DIR]:
        for txt_file in sorted(os.listdir(landmarks_dir)):
            if not txt_file.endswith('.txt'):
                continue

            txt_path = os.path.join(landmarks_dir, txt_file)
            points = []
            inside = False

            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == '{':
                        inside = True
                        continue
                    if line == '}':
                        break
                    if inside and line:
                        x, y = line.split()
                        points.append([float(x), float(y)])

            if len(points) < num_landmarks:
                print(f"{txt_file}: expected {num_landmarks}, got {len(points)}, skipping")
                continue

            img_name = txt_file.replace('.txt', '.jpg')
            all_annotations[img_name] = np.array(points[:num_landmarks], dtype=np.float32)

    return all_annotations


def make_gaussian_heatmap(landmarks, height, width, sigma=2.5):
    num_landmarks = landmarks.shape[0]
    heatmaps = np.zeros((num_landmarks, height, width), dtype=np.float32)
    landmarks_px = landmarks * np.array([width - 1, height - 1])
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    
    for i, (x, y) in enumerate(landmarks_px):
        dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
        heatmaps[i] = np.exp(-dist_sq / (2 * sigma ** 2))
    
    return heatmaps

class InfantEarDataset(Dataset):
    def __init__(self, image_files, annotations, augment=False, input_size=368, heatmap_size=23):
        """
        image_files: list of image filenames
        annotations: dict of filename -> (num_landmarks, 2)
        """
        self.images_dir = IMAGES_DIR
        self.augment = augment
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.image_files = image_files
        self.annotations = annotations

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        img = cv2.imread(img_path)

        h, w = img.shape[:2]
        landmarks = self.annotations[img_name].copy()

        if self.augment:
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)
                beta = random.randint(-20, 20)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                landmarks[:, 0] = w - landmarks[:, 0]

        landmarks[:, 0] /= w
        landmarks[:, 1] /= h
        landmarks = np.clip(landmarks, 0, 1)

        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_resized = img_resized / 255.0
        img_chw = np.transpose(img_resized, (2, 0, 1))
        heatmaps = make_gaussian_heatmap(landmarks, self.heatmap_size, self.heatmap_size, sigma=2.5)

        return (
            torch.tensor(img_chw, dtype=torch.float32),
            torch.tensor(heatmaps, dtype=torch.float32)
        )


def get_train_test_split(test_ratio=0.2, seed=42, num_landmarks=23):
    annotations = load_all_annotations(num_landmarks)
    all_filenames = sorted(annotations.keys())

    random.seed(seed)
    random.shuffle(all_filenames)

    split_idx = int(len(all_filenames) * (1 - test_ratio))
    train_files = all_filenames[:split_idx]
    test_files = all_filenames[split_idx:]

    print(f"Total labeled images: {len(all_filenames)}")
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")

    train_dataset = InfantEarDataset(train_files, annotations, augment=True)
    test_dataset = InfantEarDataset(test_files,  annotations, augment=False)

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = get_train_test_split(num_landmarks=23)
    img, heatmaps = train_dataset[0]
    
    print(f"Image shape: {img.shape}")
    print(f"Heatmaps shape: {heatmaps.shape}")
    print(f"Image min/max: {img.min():.3f} / {img.max():.3f}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
