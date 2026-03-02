import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

def detect_ear_abnormalities(image):
    # TODO: update this function to perform actual detection using model and draw on the image copy
    if image is None:
        return None
    
    img_copy = image.copy()
    
    return img_copy

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