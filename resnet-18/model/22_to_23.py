import torch
import gradio.model as model

PRETRAINED_CKPT = "infant_ear_model_best.pth"
OUTPUT_CKPT = "infant_ear_model_23lm_init.pth"

NUM_LANDMARKS_OLD = 22
NUM_LANDMARKS_NEW = 23
NUM_STAGES = 6

old_state = torch.load(PRETRAINED_CKPT, map_location="cpu")

new_model = model.get_model(NUM_LANDMARKS_NEW, NUM_STAGES)
new_state = new_model.state_dict()

old_indices = [LANDMARK_MAPPING[i] for i in range(NUM_LANDMARKS_NEW)]

transferred = []
kept_as_is  = []

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
        new_state[key][:NUM_LANDMARKS_OLD] = old_tensor.clone()
        transferred.append(key)

    elif old_tensor.shape[1] == NUM_LANDMARKS_OLD and new_tensor.shape[1] == NUM_LANDMARKS_NEW:
        new_state[key][:, :NUM_LANDMARKS_OLD, ...] = old_tensor.clone()
        transferred.append(key)

new_model.load_state_dict(new_state)