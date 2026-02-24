import csv
import math
import sys
from pathlib import Path
# NOTE: the measurements are normalized to be ratios in comparison to the height of the ear

def parse_landmarks(landmark_str):
    """Parse space-separated x y coordinate pairs into list of (x, y) tuples."""
    parts = landmark_str.strip().split(', ')
    landmarks = []
    for part in parts:
        coords = part.strip().split()
        if len(coords) == 2:
            x, y = float(coords[0]), float(coords[1])
            landmarks.append((x, y))
    return landmarks


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by vectors p2->p1 and p2->p3 in degrees."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def calculate_arc_length(landmarks, start_idx, end_idx):
    """Calculate total arc length between consecutive points from start to end."""
    total_length = 0
    for i in range(start_idx, end_idx):
        total_length += euclidean_distance(landmarks[i], landmarks[i + 1])
    return total_length


def calculate_curvature_angle(landmarks):
    """Calculate curvature angle using start (0), midpoint (9/10), and end (19) of curve."""
    start = landmarks[0]
    mid = landmarks[9]  # Midpoint of 0-19 curve
    end = landmarks[19]
    return calculate_angle(start, mid, end)


def process_csv(input_path, output_path):
    """Process input CSV and write measurements to output CSV."""
    results = []
    
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            
            image_path = row[0].strip()
            landmark_str = ', '.join(row[1:])
            landmarks = parse_landmarks(landmark_str)
            
            if len(landmarks) < 55:
                print(f"Warning: {image_path} has only {len(landmarks)} landmarks, skipping")
                continue
            
            # Calculate raw distances
            dist_0_19 = euclidean_distance(landmarks[0], landmarks[19])
            dist_4_17 = euclidean_distance(landmarks[4], landmarks[17])
            dist_25_37 = euclidean_distance(landmarks[25], landmarks[37])
            dist_0_9 = euclidean_distance(landmarks[0], landmarks[9])
            
            # Normalize using dist_0_19 as reference (ear height)
            ref_distance = dist_0_19 if dist_0_19 > 0 else 1
            
            norm_dist_0_19 = 1.0  # Reference is always 1
            norm_dist_4_17 = dist_4_17 / ref_distance
            norm_dist_25_37 = dist_25_37 / ref_distance
            norm_dist_0_9 = dist_0_9 / ref_distance
            
            # Curvature statistics (points 0 to 19)
            curvature_angle = calculate_curvature_angle(landmarks)
            arc_length = calculate_arc_length(landmarks, 0, 19)
            norm_arc_length = arc_length / ref_distance
            
            # Chord-to-arc ratio (straightness measure)
            chord_arc_ratio = dist_0_19 / arc_length if arc_length > 0 else 0
            
            results.append({
                'image_path': image_path,
                'dist_0_19_raw': dist_0_19,
                'dist_4_17_raw': dist_4_17,
                'dist_25_37_raw': dist_25_37,
                'dist_0_9_raw': dist_0_9,
                'dist_0_19_norm': norm_dist_0_19,
                'dist_4_17_norm': norm_dist_4_17,
                'dist_25_37_norm': norm_dist_25_37,
                'dist_0_9_norm': norm_dist_0_9,
                'curvature_angle_deg': curvature_angle,
                'arc_length_raw': arc_length,
                'arc_length_norm': norm_arc_length,
                'chord_arc_ratio': chord_arc_ratio
            })
    
    # Write output CSV
    fieldnames = [
        'image_path',
        'dist_0_19_raw', 'dist_4_17_raw', 'dist_25_37_raw', 'dist_0_9_raw',
        'dist_0_19_norm', 'dist_4_17_norm', 'dist_25_37_norm', 'dist_0_9_norm',
        'curvature_angle_deg', 'arc_length_raw', 'arc_length_norm', 'chord_arc_ratio'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Processed {len(results)} images. Output saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = '/home/UFAD/angelali/ears/model_output_example.csv'
        output_file = '/home/UFAD/angelali/ears/measurements_output.csv'
    
    process_csv(input_file, output_file)
