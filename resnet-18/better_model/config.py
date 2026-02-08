# config.py
# Central configuration file for paths and hyperparameters

import os

# Dataset paths
TRAIN_IMG_DIR = "../dataset/train"
TEST_IMG_DIR  = "../dataset/test"

# Number of landmarks in .pts files
NUM_LANDMARKS = 55

# Network input size (EfficientNet likes 224 or 256)
IMG_SIZE = 256

# Training settings
BATCH_SIZE = 64        # A6000 can handle large batch sizes
EPOCHS = 200            # More epochs for small dataset
LR = 1e-4               # Conservative learning rate for fine-tuning

# Use GPU
DEVICE = "cuda"

# Mixed precision speeds training on A6000
USE_AMP = True

# Output folder
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
