import json
import os

# Directories/files
LABELS_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/labels"
OUTPUT_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/landmarks"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
JSON_FILES_LABELS = [
    "0001-0010.json",
    "0011-0020.json",
    "0021-0030.json",
    "0031-0050.json",
]

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR,  exist_ok=True)

train = range(1, 31)
test = range(31, 51)

for file in JSON_FILES_LABELS:
    with open(os.path.join(LABELS_DIR, file)) as f:
        labels = json.load(f)

    for entry in labels.values():
        filename = entry["filename"]
        stem = os.path.splitext(filename)[0]
        number = int(stem.split("_")[0])
        side = stem.split("_")[1]

        if number in train:
            prefix = "train"
            output_dir = TRAIN_DIR
        elif number in test:
            prefix = "test"
            output_dir = TEST_DIR
        else:
            print(f"ERROR: {filename} is not in testing or training range!")

        landmarks = [(r["shape_attributes"]["cx"], r["shape_attributes"]["cy"])
                     for r in entry["regions"]
                     if r["shape_attributes"]["name"] == "point"]

        # Create file in same format as this dataset: https://www.dropbox.com/scl/fo/v544dqjmv5ti960p9lt51/ALrRps5ancIav_lQBY4B2qw?rlkey=7crct8zov0riqgpjjzd9swhnw&e=1&dl=0
        lines = [
            "version: 1",
            f"n_points: {len(landmarks)}",
            "{",
            *[f"{x} {y}" for x, y in landmarks],
            "}"
        ]

        file_name = f"{prefix}_{number}_{side}.txt"
        with open(os.path.join(output_dir, file_name), "w") as f:
            f.write("\n".join(lines) + "\n")