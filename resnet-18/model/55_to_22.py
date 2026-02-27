import torch
import model

PRETRAINED_CKPT = "ear_landmark_model_best.pth"
OUTPUT_CKPT= "infant_ear_model_init.pth"

NUM_LANDMARKS_OLD = 55
NUM_LANDMARKS_NEW = 22
NUM_STAGES = 6

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

old_model = model.get_model(NUM_LANDMARKS_OLD, NUM_STAGES)
checkpoint = torch.load(PRETRAINED_CKPT, map_location="cpu")

old_model.load_state_dict(checkpoint["model_state_dict"])

new_model = model.get_model(NUM_LANDMARKS_NEW, NUM_STAGES)
new_state = new_model.state_dict()

old_indices = [LANDMARK_MAPPING[i] for i in range(NUM_LANDMARKS_NEW)]

transferred = []
kept_as_is = []

for key in old_state:
    if key not in new_state:
        continue

    old_tensor = old_state[key]
    new_tensor = new_state[key]

    # If shapes match, copy directly
    if old_tensor.shape == new_tensor.shape:
        new_state[key] = old_tensor.clone()
        kept_as_is.append(key)

    elif old_tensor.shape[0] == NUM_LANDMARKS_OLD and new_tensor.shape[0] == NUM_LANDMARKS_NEW:
        new_state[key] = old_tensor[old_indices].clone()
        transferred.append(key)

    elif old_tensor.shape[1] == NUM_LANDMARKS_OLD and new_tensor.shape[1] == NUM_LANDMARKS_NEW:
        n_features = old_tensor.shape[1] - NUM_LANDMARKS_OLD
        feature_weights = old_tensor[:, :n_features, ...].clone()
        landmark_weights = old_tensor[:, n_features:, ...][:, old_indices, ...].clone()
        new_state[key] = torch.cat([feature_weights, landmark_weights], dim=1)
        transferred.append(key)

new_model.load_state_dict(new_state)

torch.save(new_model.state_dict(), OUTPUT_CKPT)