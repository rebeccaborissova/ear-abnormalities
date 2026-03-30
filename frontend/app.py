import gradio as gr
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import os
import cv2
from model import MultiStageHeatmapModel as EarLandmarkModel, soft_argmax_2d

# TO RUN: python app.py

os.environ['GRADIO_TEMP_DIR'] = os.path.expanduser('~/gradio_tmp')
os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)

model = EarLandmarkModel(num_landmarks=23).to("cpu")
model.load_state_dict(
    torch.load(f"infant_ear_model_23lm_best_v2.pth", map_location="cpu")
)
model.eval()

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
    
    # denormalize to original image space
    pred_px = pred * np.array([orig_w - 1, orig_h - 1])
    
    img_cv = cv2.cvtColor(np.array(img_copy), cv2.COLOR_RGB2BGR)
    
    # draw the landmarks (red circles)
    for (x, y) in pred_px:
        cv2.circle(img_cv, (int(x), int(y)), 6, (0, 0, 255), -1)
    
    # draw lines (green lines)
    line_pairs = [(0, 19), (0, 8), (4, 17), (21, 22), (20, 21)]
    for (i, j) in line_pairs:
        pt1 = (int(pred_px[i][0]), int(pred_px[i][1]))
        pt2 = (int(pred_px[j][0]), int(pred_px[j][1]))
        cv2.line(img_cv, pt1, pt2, (0, 255, 0), 2)
    
    img_result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    return img_result

with gr.Blocks(css="""
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .input-column {
        border-right: 2px solid #e0e0e0;
        padding-right: 20px;
    }
    .output-column {
        padding-left: 20px;
    }
""") as demo:
    gr.Markdown("# Ear abnormality detection and severity assessment")
    
    with gr.Row():
        with gr.Column(elem_classes="input-column"):
            gr.Markdown("### Input Image")
            input_image = gr.Image(
                type="pil",
                label="Upload Ear Image",
                height=400
            )
            submit_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column(elem_classes="output-column"):
            gr.Markdown("### Detection Results")
            output_image = gr.Image(
                type="pil",
                label="Labeled Image",
                height=400
            )
    
    submit_btn.click(
        fn=detect_ear_abnormalities,
        inputs=input_image,
        outputs=output_image,
        api_name="predict"
    )

demo.launch(share=True)