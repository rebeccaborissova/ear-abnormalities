# model.py

import torch.nn as nn
import torchvision.models as models
from config import *

class EarLandmarkModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, NUM_LANDMARKS * 2)
        )

    def forward(self, x):
        return self.backbone(x)
