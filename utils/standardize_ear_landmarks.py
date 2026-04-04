import json
import os

# Directories/files
LABELS_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/labels"
OUTPUT_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/landmarks/sus_amongus"
JSON_FILES_LABELS = [
    "0001-0010.json",
    "0011-0020.json",
    "0021-0030.json",
    "0031-0050.json",
    "0100-0150.json",
    "category 2 over 60.json",
    "4_score_ears.json"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in JSON_FILES_LABELS:
    file_path = os.path.join(LABELS_DIR, file)
    with open(file_path) as f:
        labels = json.load(f)

    entries = labels["_via_img_metadata"].values() if "_via_img_metadata" in labels else labels.values()

    for entry in entries:
        filename = entry["filename"]

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

        file_name = filename.replace(".jpg", ".txt")
        with open(os.path.join(OUTPUT_DIR, file_name), "w") as f:
            f.write("\n".join(lines) + "\n")