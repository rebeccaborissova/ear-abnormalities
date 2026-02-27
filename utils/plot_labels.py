import os
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image

# Directories/files
LANDMARKS_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/landmarks"
IMG_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/images"
OUTPUT_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/labeled_ears"

os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test"),  exist_ok=True)

def read_txt(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    start = lines.index('{\n') + 1
    end = lines.index('}\n')

    points = []
    for l in lines[start:end]:
        x, y = map(float, l.strip().split())
        points.append((x, y))
    return points

def plot_labels(landmarks_dir = LANDMARKS_DIR, img_dir = IMG_DIR, output_dir = OUTPUT_DIR):
    for split in ["train", "test"]:
        landmarks_dir_split = os.path.join(landmarks_dir, split)
        output_dir_split = os.path.join(output_dir, split)

        for landmarks_file in sorted(os.listdir(landmarks_dir_split)):
            stem = os.path.splitext(landmarks_file)[0]
            parts = stem.split("_")
            
            number = parts[0]
            side = parts[1]

            img_name = f"{number}_{side}.jpg"

            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                print(f"ERROR: Image not found for {landmarks_file}!")

            landmarks = read_txt(os.path.join(landmarks_dir_split, landmarks_file))
            colors = [hsv_to_rgb([i / len(landmarks), 0.85, 0.95]) for i in range(len(landmarks))]

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(Image.open(img_path).convert("RGB"))

            for i, (cx, cy) in enumerate(landmarks):
                ax.plot(cx, cy, "o", color=colors[i], markeredgecolor="white")
                ax.text(cx + 6, cy - 6, str(i + 1), color=colors[i], fontsize=8, fontweight="bold")

            labeled_img_path = os.path.join(output_dir_split, landmarks_file.replace(".txt", ".png"))
            
            ax.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(labeled_img_path)
            plt.close()

if __name__ == "__main__":
    plot_labels()