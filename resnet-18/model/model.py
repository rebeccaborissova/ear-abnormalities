# model.py
# This file defines the neural network architecture.

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important regions"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class LandmarkRefinementHead(nn.Module):
    """Custom head for landmark detection with spatial refinement"""
    def __init__(self, in_channels, num_landmarks):
        super(LandmarkRefinementHead, self).__init__()
        
        # Multi-scale feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        self.attention1 = SpatialAttention(512)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        self.attention2 = SpatialAttention(256)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with residual-like connections
        self.fc1 = nn.Linear(64, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.3)
        
        # Output layer
        self.fc_out = nn.Linear(128, num_landmarks * 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Multi-scale feature extraction with attention
        x = self.conv1(x)
        x = self.attention1(x)
        
        x = self.conv2(x)
        x = self.attention2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # FC layers with normalization
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc_out(x)
        x = self.sigmoid(x)
        
        return x

def get_model(num_landmarks=55):
    """
    Load pretrained ResNet-18 as backbone and add enhanced landmark detection head.
    """
    # Load pretrained ResNet-18 as backbone
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Remove the final FC layer and avgpool
    modules = list(resnet.children())[:-2]  # Remove avgpool and fc
    backbone = nn.Sequential(*modules)
    
    # Create model with enhanced head
    class EarLandmarkModel(nn.Module):
        def __init__(self, backbone, num_landmarks):
            super(EarLandmarkModel, self).__init__()
            self.backbone = backbone
            self.head = LandmarkRefinementHead(512, num_landmarks)
        
        def forward(self, x):
            x = self.backbone(x)
            x = self.head(x)
            return x
    
    model = EarLandmarkModel(backbone, num_landmarks)
    
    return model
