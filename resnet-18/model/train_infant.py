import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import model
from infant_dataset import get_train_test_split

NUM_LANDMARKS = 22

PRETRAINED_CKPT = f"infant_ear_model_{NUM_LANDMARKS}lm_init.pth"
OUTPUT_CKPT = f"infant_ear_model_{NUM_LANDMARKS}lm_best.pth"

NUM_STAGES = 6
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = model.get_model(NUM_LANDMARKS, NUM_STAGES).to(device)
model.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=device))

train_dataset, test_dataset = get_train_test_split(num_landmarks=NUM_LANDMARKS)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, target_heatmaps in train_loader:
        imgs = imgs.to(device)
        target_heatmaps = target_heatmaps.to(device)

        optimizer.zero_grad()
        stage_outputs = model(imgs)

        loss = 0.0
        for s in range(NUM_STAGES):
            loss += criterion(stage_outputs[:, s], target_heatmaps)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for imgs, target_heatmaps in test_loader:
            imgs = imgs.to(device)
            target_heatmaps = target_heatmaps.to(device)

            stage_outputs = model(imgs)
            loss = criterion(stage_outputs[:, -1], target_heatmaps)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    scheduler.step(val_loss)

    print(f"Epoch #{epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss} - Val Loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), OUTPUT_CKPT)
        print(f"--> Saved new checkpoint! Val loss: {val_loss}")

print(f"Final val loss: {best_val_loss}")