# utils.py
# Utility functions for parsing landmark files

import numpy as np

def load_pts(file_path):
    """
    Read .pts file and return Nx2 landmarks.
    """

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find landmark block
    start = lines.index("{\n") + 1
    end = lines.index("}\n")

    points = []
    for line in lines[start:end]:
        x, y = line.strip().split()
        points.append([float(x), float(y)])

    return np.array(points, dtype=np.float32)
