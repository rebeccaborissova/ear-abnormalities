import gradio as gr
import numpy as np
from PIL import Image
import torch
import os
import cv2
import math
from model import MultiStageHeatmapModel as EarLandmarkModel, soft_argmax_2d

# TO RUN: python app.py

os.environ['GRADIO_TEMP_DIR'] = os.path.expanduser('~/gradio_tmp')
os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)

model = EarLandmarkModel(num_landmarks=23).to("cpu")
model.load_state_dict(
    torch.load("infant_ear_model_23lm_best_v2.pth", map_location="cpu")
)
model.eval()

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

LANDMARK_GROUPS = [
    (range(0,  4),  (180, 220, 255), "#B4DCFF", "Ascending Helix (0-3)"),
    (range(4,  8),  (255, 220,   0), "#FFDC00", "Descending Helix (4-7)"),
    (range(8, 14),  (255, 140,   0), "#FF8C00", "Helix (8-13)"),
    (range(14, 20), (255,  80,  80), "#FF5050", "Ear Lobe (14-19)"),
]

NAMED_LANDMARKS = {
    20: ("Tragus",                 (  0, 200, 255), "#00C8FF"),
    21: ("Superior Helix",         (  0, 255, 100), "#00FF64"),
    22: ("Cartilage Intersection", (200,   0, 255), "#C800FF"),
}

# Measurement lines
LINE_PAIRS = [
    (0,  19, (255,  50,  50),  "#FF3232", "Ear Height"),
    (0,   8, (255, 165,   0),  "#FFA500", "Upper Height"),
    (4,  17, (255, 220,   0),  "#FFDC00", "Ear Width"),
    (20, 21, (  0, 180, 255),  "#00B4FF", "Inner Width"),
    (21, 22, (  0, 255, 100),  "#00FF64", "Inner Height"),
]

def get_landmark_color(i):
    for group_range, color, _, _ in LANDMARK_GROUPS:
        if i in group_range:
            return color
    if i in NAMED_LANDMARKS:
        return NAMED_LANDMARKS[i][1]
    return (220, 220, 220)

def detect_ear_abnormalities(image):
    if image is None:
        return None, ""

    img_copy  = image.copy()
    orig_w, orig_h = img_copy.size

    img_resized    = img_copy.resize((368, 368), Image.BILINEAR)
    img_normalized = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor     = torch.tensor(
        np.transpose(img_normalized, (2, 0, 1)), dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        all_stages     = model(img_tensor)
        final_heatmaps = all_stages[:, -1, :, :, :]
        coords         = soft_argmax_2d(final_heatmaps, normalize=True)

    pred    = coords[0].cpu().numpy()
    pred_px = pred * np.array([orig_w - 1, orig_h - 1])

    img_cv = cv2.cvtColor(np.array(img_copy), cv2.COLOR_RGB2BGR)

    # draw measurement lines
    for (i, j, bgr, _, _label) in LINE_PAIRS:
        pt1 = (int(pred_px[i][0]), int(pred_px[i][1]))
        pt2 = (int(pred_px[j][0]), int(pred_px[j][1]))
        cv2.line(img_cv, pt1, pt2, bgr, 3, cv2.LINE_AA)

    # draw all landmarks
    for i, (x, y) in enumerate(pred_px):
        color = get_landmark_color(i)
        size  = 9 if i in NAMED_LANDMARKS else 6
        cv2.circle(img_cv, (int(x), int(y)), size, color, -1)
        cv2.circle(img_cv, (int(x), int(y)), size, (0, 0, 0), 1)

    img_result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    ear_height = dist(pred_px[0], pred_px[19])
    ref = ear_height if ear_height > 0 else 1

    # legend
    legend_html = "<div style='display:grid; grid-template-columns:1fr 1fr; gap:8px 24px; margin-bottom:16px;'>"
    for _, _, hex_color, label in LANDMARK_GROUPS:
        legend_html += f"<span><span style='color:{hex_color}; font-size:18px;'>●</span>&nbsp;{label}</span>"
    for idx, (name, _, hex_color) in NAMED_LANDMARKS.items():
        legend_html += f"<span><span style='color:{hex_color}; font-size:18px;'>●</span>&nbsp;Pt {idx}: {name}</span>"
    legend_html += "</div>"

    # statistics table
    table_html = "<table style='width:100%; border-collapse:collapse;'>"
    for (i, j, _, hex_color, label) in LINE_PAIRS:
        val = f"{dist(pred_px[i], pred_px[j]) / ref:.3f}"
        table_html += f"""
        <tr style='border-bottom:1px solid #eee;'>
            <td style='padding:6px 8px;'>
                <span style='color:{hex_color}; font-size:18px;'>━</span>
                <span style='margin-left:8px;'>{label}</span>
            </td>
            <td style='padding:6px 8px; text-align:right; font-weight:bold;'>{val}</td>
        </tr>"""
    table_html += "</table>"

    return img_result, legend_html + table_html


with gr.Blocks() as demo:
    gr.Markdown("# Ear Abnormality Detection and Severity Assessment")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Image")
            input_image = gr.Image(type="pil", label="Upload Ear Image", height=400)
            submit_btn  = gr.Button("Analyze", variant="primary")

        with gr.Column():
            gr.Markdown("### Detection Results")
            output_image = gr.Image(type="pil", label="Labeled Image", height=400)
            output_html  = gr.HTML(label="Measurements")

    submit_btn.click(
        fn=detect_ear_abnormalities,
        inputs=input_image,
        outputs=[output_image, output_html],
        api_name="predict"
    )

demo.launch(share=True)