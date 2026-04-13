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
from ml_model.infant_confidence_utils import (
    get_confidence_for_landmarks,
    get_total_confidence,
    draw_landmarks_with_confidence,
    get_confidence_color,
)

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

def detect_ear_abnormalities(image):
    if image is None:
        return None
    
    img_copy = image.copy()
    orig_w, orig_h = img_copy.size
    
    img_resized = img_copy.resize((368, 368), Image.BILINEAR)
    
    # normalize coordinates
    img_normalized = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0).to("cpu")
    
    with torch.no_grad():
        all_stages = model(img_tensor)
        final_heatmaps = all_stages[:, -1, :, :, :]
        coords = soft_argmax_2d(final_heatmaps, normalize=True)
    
    pred = coords[0].cpu().numpy()
    confidences = get_confidence_for_landmarks(final_heatmaps[0].cpu().numpy())
    total_confidence = get_total_confidence(confidences, metric='mean')
    
    # denormalize to original image space
    pred_px = pred * np.array([orig_w - 1, orig_h - 1])
    
    img_cv = cv2.cvtColor(np.array(img_copy), cv2.COLOR_RGB2BGR)

    # draw measurement lines
    line_pairs = [
        (0,  19, (255, 50,  50),  "Ear Height"),
        (0,  8,  (255, 165,  0),  "Upper Height"),
        (4,  17, (255, 255,  0),  "Ear Width"),
        (20, 21, (0,  180, 255),  "Inner Width"),
        (21, 22, (0,  255, 100), "Inner Height"),
    ]

    ear_height = dist(pred_px[0], pred_px[19])
    ref = ear_height if ear_height > 0 else 1

    for (i, j, color, label) in line_pairs:
        pt1 = (int(pred_px[i][0]), int(pred_px[i][1]))
        pt2 = (int(pred_px[j][0]), int(pred_px[j][1]))
        cv2.line(img_cv, pt1, pt2, color, 3, cv2.LINE_AA)

    img_cv = draw_landmarks_with_confidence(
        img_cv,
        pred_px,
        confidences,
        high_conf_threshold=0.8,
        medium_conf_threshold=0.6,
        low_conf_threshold=0.4,
        confidence_style='distinct',
        show_confidence_text=True,
        label_only_index=True,
    )

    img_result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    line_pairs_hex = [
        (0,  19, "#32C8FF", "Ear Height"),
        (0,  8,  "#FFC832", "Upper Height"),
        (4,  17, "#50C850", "Ear Width"),
        (20, 21, "#00C8FF", "Inner Width"),
        (21, 22, "#C800FF", "Inner Height"),
    ]

    legend_items = [
        ("#FFFFFF", "Pts 0–19: Helix outline"),
        ("#FFB400", "Pt 20: Tragus"),
        ("#00FF64", "Pt 21: Superior helix"),
        ("#C800FF", "Pt 22: Cartilage intersection"),
    ]
    
    legend_html = "<div style='display:grid; grid-template-columns:1fr 1fr; gap:8px 24px; margin-bottom:16px;'>"
    for color, name in legend_items:
        legend_html += f"<span><span style='color:{color}; font-size:18px;'>●</span> {name}</span>"
    legend_html += "</div>"

    # confidence summary
    high_count = int(np.sum(confidences >= 0.8))
    medium_count = int(np.sum((confidences >= 0.6) & (confidences < 0.8)))
    low_count = int(np.sum( confidences < 0.6))
    

    confidence_legend_items = [
        ("#00FF00", "High confidence ≥ 0.8"),
        ("#FFFF00", "Medium confidence ≥ 0.6"),
        ("#FF0000", "Low confidence < 0.6"),
    ]
    confidence_legend_html = "<div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin:12px 0 16px;'>"
    for color, name in confidence_legend_items:
        confidence_legend_html += f"<span><span style='color:{color}; font-size:18px;'>●</span> {name}</span>"
    confidence_legend_html += "</div>"

    per_landmark_html = "<div style='display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:6px;'>"
    for idx, conf in enumerate(confidences):
        color = get_confidence_color(conf, confidence_style='distinct')
        hex_color = f"#{color[2]:02X}{color[1]:02X}{color[0]:02X}"
        per_landmark_html += (
            f"<span style='display:flex; align-items:center; gap:6px; white-space:nowrap;'>"
            f"<span style='color:{hex_color}; font-size:16px;'>●</span>"
            f"Pt {idx}: {conf:.2f}</span>"
        )
    per_landmark_html += "</div>"

    confidence_html = f"""
    <div style='margin-top:16px;'>
        <h4 style='margin-bottom:8px;'>Confidence Scoring</h4>
        <p style='margin:0 0 8px;'>Total confidence (mean): <strong>{total_confidence:.3f}</strong></p>
        <div style='display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:6px; margin-bottom:12px;'>
            <span>High: <strong>{high_count}</strong></span>
            <span>Medium: <strong>{medium_count}</strong></span>
            <span>Low: <strong>{low_count}</strong></span>
        </div>
        {confidence_legend_html}
        <div style='font-size:14px; line-height:1.4; color:#333;'>
            <strong>Per-landmark scores</strong>
        </div>
        {per_landmark_html}
    </div>
    """

    # statistics table
    table_html = "<table style='width:100%; border-collapse:collapse;'>"
    for (i, j, hex_color, label) in line_pairs_hex:
        d = dist(pred_px[i], pred_px[j])
        val = f"{d/ref:.3f}"
        table_html += f"""
        <tr style='border-bottom:1px solid #eee;'>
            <td style='padding:6px 8px;'>
                <span style='color:{hex_color}; font-size:18px;'>━</span>
                <span style='margin-left:8px;'>{label}</span>
            </td>
            <td style='padding:6px 8px; text-align:right; font-weight:bold;'>{val}</td>
        </tr>"""
    table_html += "</table>"
    html_output = legend_html + table_html + confidence_html

    return img_result, html_output


with gr.Blocks() as demo:
    gr.Markdown("# Ear Abnormality Detection and Severity Assessment")
    
    with gr.Row():
        with gr.Column(elem_classes="input-column"):
            gr.Markdown("### Input Image")
            input_image = gr.Image(type="pil", label="Upload Ear Image", height=400)
            submit_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column(elem_classes="output-column"):
            gr.Markdown("### Detection Results")
            output_image = gr.Image(type="pil", label="Labeled Image", height=400)
            output_html = gr.HTML(label="Measurements")
    
    submit_btn.click(
        fn=detect_ear_abnormalities,
        inputs=input_image,
        outputs=[output_image, output_html],
        api_name="predict"
    )

demo.launch(share=True)