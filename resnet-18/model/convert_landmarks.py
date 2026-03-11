import os
import glob

LANDMARKS_DIR = "/home/UFAD/rborissova/senior_project/BabyEar4k/landmarks/train"
OUTPUT_CSV    = "/home/UFAD/rborissova/senior_project/ear-abnormalities/measurements/landmarks.csv"

def parse_pts_file(filepath):
    points = []
    inside = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '{':
                inside = True
                continue
            if line == '}':
                break
            if inside and line:
                x, y = line.split()
                points.append((float(x), float(y)))
    return points

def convert_landmarks_to_csv(landmarks_dir, output_csv):
    txt_files = sorted(glob.glob(os.path.join(landmarks_dir, "*.txt")))
    rows = []

    for txt_path in txt_files:
        points = parse_pts_file(txt_path)
        if len(points) != 23:
            print(f"Warning: {txt_path} has {len(points)} points, skipping")
            continue

        basename = os.path.basename(txt_path).replace('.txt', '.jpg')
        coords   = ', '.join(f"{x} {y}" for x, y in points)
        rows.append(f"{basename}, {coords}")

    with open(output_csv, 'w') as f:
        f.write('\n'.join(rows))

    print(f"Wrote {len(rows)} rows to {output_csv}")

if __name__ == "__main__":
    convert_landmarks_to_csv(LANDMARKS_DIR, OUTPUT_CSV)