# ear-abnormalities
Public demo link: https://huggingface.co/spaces/rebeccassova/Infant_Ear_Abnormalities

## Overview 
This project develops a machine learning–based framework for objective assessment of infant ear deformities. Current diagnosis in pediatric plastic surgery is largely subjective. Our goal is to build a standardized, quantitative pipeline that:
- Detects ear landmarks automatically
- Computes geometric measurements (distances, ratios, curvature)
- Predicts deformity types

Instead of treating deformity detection as a black-box classification problem, we focus on measurement-driven and clinically interpretable modeling.

<img width="1190" height="412" alt="image" src="https://github.com/user-attachments/assets/e46e8e42-9b7d-4d1a-a4b7-5f0385be81ef" />
<img width="1189" height="705" alt="image" src="https://github.com/user-attachments/assets/5464bf10-95d6-442d-905b-05b59ad3da95" />

As shown by the above screenshots, once the user uploads an image, 23 ear landmarks are predicted and mapped. The left side of the screen shows the confidence levels the model has for each landmark. The right side shows the landmarks, calculated lines and measurements, and predicted diagnosis. 

## Technologies
- Python
- PyTorch
- TorchVision
- NumPy
- Gradio

## How to Train the Ear Landmark Prediction model
1. Clone the main branch of this repository. 
2. Download the adult ear data from [iBug](https://ibug.doc.ic.ac.uk/resources/ibug-ears/). (If iBug link is down, use this [Dropbox link](http://dropbox.com/scl/fo/v544dqjmv5ti960p9lt51/ALrRps5ancIav_lQBY4B2qw?rlkey=7crct8zov0riqgpjjzd9swhnw&dl=0)). 
3. Download the infant ear data from [BabyEar4k](https://doi.org/10.6084/m9.figshare.c.6841596.v1).
4. In the terminal, go to the `backend/landmark_model` folder. 
5. Update the `config.yaml` file, specifically considering the variables that are labeled “UPDATE”, such as file paths of the downloaded dataset.
6. Run `python pipeline.py` to run the full pipeline, starting to train the adult ear model first and then the infant ear model. If the adult ear model is already trained, you can skip the adult model steps by running `python pipeline.py –skip adult`. See further configurable options in the `README.md` in that folder.


## Credits
The final model found in the **backend** folder is based on an existing transfer learning model from [**resnet-18**](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html). We replaced the classification head of the model and replaced it with a regression head. We then further replaced direct coordinate regression with a multi-stage heatmap-based landmark localization architecture.

To train the model on adult ears, we used data made available through [iBug](https://ibug.doc.ic.ac.uk/resources/ibug-ears/) (also available at this [Dropbox link](http://dropbox.com/scl/fo/v544dqjmv5ti960p9lt51/ALrRps5ancIav_lQBY4B2qw?rlkey=7crct8zov0riqgpjjzd9swhnw&dl=0)).

To train the model on infant ears, we worked on manually labeling this dataset of [infant ears](https://www.nature.com/articles/s41597-023-02834-4). 23 points were labeled on each of the ear images, specifically #0-19, 37, 26, and a new point between 48 and 53 on the intersection of the cartilage. This dataset has not been labelled before for ear landmarks.

<img width="240" height="210" alt="IMG_0251" src="https://github.com/user-attachments/assets/d553c1b0-f0ef-4fa8-ba09-e39532559aaf" />
