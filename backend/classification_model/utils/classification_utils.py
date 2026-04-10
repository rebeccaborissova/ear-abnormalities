import os
import yaml

def load_config():
    with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
        return yaml.safe_load(f)

def parse_label(label_str):
    try:
        parts = str(label_str).strip().split('+')
        if len(parts) != 3:
            return None
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, AttributeError):
        return None

def extract_side(img_name):
    if len(img_name) > 5 and img_name[5] in ("L", "R"):
        return img_name[5]
    for candidate in ("_L_", "_R_", "/L/", "/R/"):
        if candidate in img_name:
            return candidate[1]
    return None