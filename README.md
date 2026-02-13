# ear-abnormalities

## Overview 
This project develops a machine learning–based framework for objective assessment of infant ear deformities. Current diagnosis in pediatric plastic surgery is largely subjective. Our goal is to build a standardized, quantitative pipeline that:
- Detects ear landmarks automatically
- Computes geometric measurements (distances, ratios, curvature)
- Generates interpretable deformity severity scores

Instead of treating deformity detection as a black-box classification problem, we focus on measurement-driven and clinically interpretable modeling.

## Credits
**ear-landmark-detection-with-CNN** folder is based on https://github.com/kbulutozler/ear-landmark-detection-with-CNN. This was an existing model. In our project, we use this repository as a baseline landmark detector. We adapt the training pipeline to our dataset and work on improving landmark localization stability.

**resnet-18** folder is based on an existing transfer learning model from [**resnet-18**](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html). We plan on replacing the classification head of the model and replacing it with a regression head.

## Technologies
- Python
- PyTorch
- NumPy / SciPy
- Gradio
