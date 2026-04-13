import gradio as gr
import numpy as np
from PIL import Image
import torch
import os
import sys
import cv2
import math
from model import MultiStageHeatmapModel as EarLandmarkModel, soft_argmax_2d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.landmark_model.infant_confidence_utils import (
    get_confidence_for_landmarks,
    get_total_confidence,
    draw_landmarks_with_confidence,
    get_confidence_color,
)

os.environ['GRADIO_TEMP_DIR'] = os.path.expanduser('~/gradio_tmp')
os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)

model = EarLandmarkModel(num_landmarks=23).to("cpu")
model.load_state_dict(
    torch.load("infant_ear_model_23lm_best_v2.pth", map_location="cpu")
)
model.eval()

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# landmark colors
LANDMARK_COLORS = {
    0:  (0,   215, 255),  # otobasion superius: gold
    4:  (100, 100, 255),  # superaurale: red
    8:  (100, 255, 100),  # postaurale: green
    17: (255, 202, 100),  # subaurale: blue
    19: (50,  150, 255),  # otobasion inferius: orange
    20: (0,   180, 255),  # darwinian tubercle: yellow
    21: (100, 255,   0),  # deepest pt above tragion: lime
    22: (255,   0, 200),  # sup & inf crus junction: purple
}
HELIX_COLOR = (220, 220, 220)

# Measurement lines: (pt_a, pt_b, bgr, hex, label, thickness)
LINE_PAIRS = [
    (21, 20, (193, 182, 255), "#FFB6C1", "Morphologic length",       5),
    (0,   8, (128,   0, 128), "#800080", "Physiognomic width",        3),
    (4,  17, (141, 155,   0), "#009B8D", "Physiognomic length",       3),
    (0,  19, (0,     0, 139), "#8B0000", "Morphologic width",         3),
    (21, 22, (80,  165,   0), "#00A550", "Helix-antihelix distance",  3),
]

LANDMARK_LEGEND = [
    ("#FFFFFF", "Pts 1–3: Helix outline"),
    ("#FFD700", "Pt 0: Otobasion superius"),
    ("#FF6464", "Pt 4: Superaurale"),
    ("#64FF64", "Pt 8: Postaurale"),
    ("#64CAFF", "Pt 17: Subaurale"),
    ("#FF9632", "Pt 19: Otobasion inferius"),
    ("#FFB400", "Pt 20: Darwinian tubercle"),
    ("#00FF64", "Pt 21: Deepest point above tragion"),
    ("#C800FF", "Pt 22: Superior and inferior crus junction"),
]


def get_landmark_color(i):
    if i in LANDMARK_COLORS:
        return LANDMARK_COLORS[i]
    return HELIX_COLOR


def detect_ear_abnormalities(image):
    if image is None:
        return None, None, "", ""

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

    pred        = coords[0].cpu().numpy()
    confidences = get_confidence_for_landmarks(final_heatmaps[0].cpu().numpy())
    total_conf  = get_total_confidence(confidences, metric='mean')
    pred_px     = pred * np.array([orig_w - 1, orig_h - 1])

    # left image: confidence colors
    img_conf = cv2.cvtColor(np.array(img_copy), cv2.COLOR_RGB2BGR)
    img_conf = draw_landmarks_with_confidence(
        img_conf, pred_px, confidences,
        high_conf_threshold=0.8,
        medium_conf_threshold=0.6,
        low_conf_threshold=0.4,
        confidence_style='distinct',
        show_confidence_text=True,
        label_only_index=True,
    )
    img_conf_pil = Image.fromarray(cv2.cvtColor(img_conf, cv2.COLOR_BGR2RGB))

    # right image: anatomical colors + measurement lines
    img_anat = cv2.cvtColor(np.array(img_copy), cv2.COLOR_RGB2BGR)

    for (i, j, bgr, _, _label, thickness) in LINE_PAIRS:
        cv2.line(img_anat,
                 (int(pred_px[i][0]), int(pred_px[i][1])),
                 (int(pred_px[j][0]), int(pred_px[j][1])),
                 bgr, thickness, cv2.LINE_AA)

    for i, (x, y) in enumerate(pred_px):
        color = get_landmark_color(i)
        size  = 9 if i in LANDMARK_COLORS else 5
        cv2.circle(img_anat, (int(x), int(y)), size, color, -1)
        cv2.circle(img_anat, (int(x), int(y)), size, (0, 0, 0), 1)

    img_anat_pil = Image.fromarray(cv2.cvtColor(img_anat, cv2.COLOR_BGR2RGB))

    # left description: confidence summary
    high_count   = int(np.sum(confidences >= 0.8))
    medium_count = int(np.sum((confidences >= 0.6) & (confidences < 0.8)))
    low_count    = int(np.sum(confidences < 0.6))

    per_landmark_html = "<div style='display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:6px; margin-top:8px;'>"
    for idx, conf in enumerate(confidences):
        color     = get_confidence_color(conf, confidence_style='distinct')
        hex_color = f"#{color[2]:02X}{color[1]:02X}{color[0]:02X}"
        per_landmark_html += (
            f"<span style='display:flex; align-items:center; gap:4px; white-space:nowrap; font-size:13px;'>"
            f"<span style='color:{hex_color}; font-size:15px;'>●</span>"
            f"Pt {idx}: {conf:.2f}</span>"
        )
    per_landmark_html += "</div>"

    conf_html = f"""
    <div>
        <p style='margin:0 0 8px;'>Total confidence (mean): <strong>{total_conf:.3f}</strong></p>
        <div style='display:grid; grid-template-columns:repeat(3, 1fr); gap:6px; margin-bottom:12px;'>
            <span>High: <strong>{high_count}</strong></span>
            <span>Medium: <strong>{medium_count}</strong></span>
            <span>Low: <strong>{low_count}</strong></span>
        </div>
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin-bottom:12px;'>
            <span><span style='color:#00FF00; font-size:15px;'>●</span> High ≥ 0.8</span>
            <span><span style='color:#FFFF00; font-size:15px;'>●</span> Medium ≥ 0.6</span>
            <span><span style='color:#FF0000; font-size:15px;'>●</span> Low &lt; 0.6</span>
        </div>
        <strong style='font-size:14px;'>Per-landmark scores</strong>
        {per_landmark_html}
    </div>
    """

    # right description: legend + measurements
    legend_html = "<div style='display:grid; grid-template-columns:1fr 1fr; gap:6px 16px; margin-bottom:16px;'>"
    for color, name in LANDMARK_LEGEND:
        legend_html += (
            f"<span style='font-size:13px;'>"
            f"<span style='color:{color}; font-size:15px;'>●</span> {name}</span>"
        )
    legend_html += "</div>"

    ref = dist(pred_px[4], pred_px[17])
    ref = ref if ref > 0 else 1

    table_html = "<table style='width:100%; border-collapse:collapse;'>"
    for (i, j, _, hex_color, label, _t) in LINE_PAIRS:
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

    anat_html = legend_html + table_html

    return img_conf_pil, img_anat_pil, conf_html, anat_html


with gr.Blocks() as demo:
    gr.Markdown("# Ear Abnormality Detection and Severity Assessment")

    input_image = gr.Image(type="pil", label="Upload Ear Image", height=400)
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            submit_btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=1):
            pass

    with gr.Row(visible=False) as results_row:
        with gr.Column():
            gr.Markdown("### Confidence Scoring")
            output_conf_img  = gr.Image(type="pil", label="Confidence Map", height=380)
            output_conf_html = gr.HTML()

        with gr.Column():
            gr.Markdown("### Landmark Detection & Measurements")
            output_anat_img  = gr.Image(type="pil", label="Labeled Image", height=380)
            output_anat_html = gr.HTML()

    def run_and_show(image):
        conf_img, anat_img, conf_html, anat_html = detect_ear_abnormalities(image)
        return (
            conf_img, anat_img, conf_html, anat_html,
            gr.update(visible=True),
        )

    submit_btn.click(
        fn=run_and_show,
        inputs=input_image,
        outputs=[output_conf_img, output_anat_img, output_conf_html, output_anat_html, results_row],
    )

demo.launch(share=True)