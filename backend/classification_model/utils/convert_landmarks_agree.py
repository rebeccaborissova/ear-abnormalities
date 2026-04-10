import os
import csv
import yaml

with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
    cfg = yaml.safe_load(f)

DIAGNOSIS_CSV = cfg['diagnosis_csv']
LANDMARKS_CSV = "../landmark_model/postprocess/landmarks.csv"
OUTPUT_CSV = "../landmark_model/postprocess/agreed_landmarks.csv"

def load_rater_agreed_ears(diagnosis_csv):
    agreeing = set()
    disagreeing = set()

    with open(diagnosis_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['L01'] == row['L02'] == row['L_merge']:
                agreeing.add(os.path.basename(row['dictory_L']))
            else:
                disagreeing.add(os.path.basename(row['dictory_L']))
            if row['R01'] == row['R02'] == row['R_merge']:
                agreeing.add(os.path.basename(row['dictory_R']))
            else:
                disagreeing.add(os.path.basename(row['dictory_R']))

    print(f"Ears with full agreement: {len(agreeing)}")
    print(f"Ears with disagreement:   {len(disagreeing)}")
    return agreeing


def filter_landmarks_csv(landmarks_csv, diagnosis_csv, output_csv):
    agreeing = load_rater_agreed_ears(diagnosis_csv)

    rows = []
    skipped = 0

    with open(landmarks_csv, 'r') as f:
        for line in f:
            basename = line.split(',')[0].strip()
            if basename in agreeing:
                rows.append(line.rstrip())
            else:
                skipped += 1

    with open(output_csv, 'w') as f:
        f.write('\n'.join(rows))

    print(f"\nWrote {len(rows)} rows to {output_csv}")
    print(f"Skipped {skipped} ears due to rater disagreement")


if __name__ == "__main__":
    filter_landmarks_csv(LANDMARKS_CSV, DIAGNOSIS_CSV, OUTPUT_CSV)