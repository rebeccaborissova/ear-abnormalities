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

To run the model, we worked on manually labeling this dataset of [infant ears](https://www.nature.com/articles/s41597-023-02834-4). 23 points were labeled on each of the ear images, specifically #0-19, 37, 26, and a new point between 48 and 53 on the intersection of the cartilage. This dataset has not been labelled before for ear landmarks.
<img width="240" height="210" alt="IMG_0251" src="https://github.com/user-attachments/assets/d553c1b0-f0ef-4fa8-ba09-e39532559aaf" />


## Technologies
- Python
- PyTorch
- NumPy / SciPy
- Gradio
