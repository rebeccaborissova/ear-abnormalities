# model.py
# Multi-stage heatmap-based model inspired by the OpenPose architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


class HeatmapStage(nn.Module):
    """Single refinement stage for heatmap prediction"""
    def __init__(self, in_channels, num_landmarks):
        super(HeatmapStage, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Output heatmap
        self.conv_out = nn.Conv2d(128, num_landmarks, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        heatmaps = self.conv_out(x)
        return heatmaps


class MultiStageHeatmapModel(nn.Module):
    """Multi-stage heatmap prediction model"""
    def __init__(self, num_landmarks=55, num_stages=6):
        super(MultiStageHeatmapModel, self).__init__()
        
        self.num_stages = num_stages
        
        # Feature extractor (ResNet-18 backbone)
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Take up to layer3 to get reasonable feature map size
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:7])
        # Output: 256 channels at 1/16 resolution
        
        # Reduce channels for initial stage
        self.initial_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Initial stage
        self.stage1 = HeatmapStage(128, num_landmarks)
        
        # Subsequent refinement stages
        self.stages = nn.ModuleList([
            HeatmapStage(128 + num_landmarks, num_landmarks) 
            for _ in range(num_stages - 1)
        ])
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        features = self.initial_conv(features)
        
        # Stage 1
        heatmap_stage1 = self.stage1(features)
        stage_outputs = [heatmap_stage1]
        
        # Refinement stages
        for stage in self.stages:
            # Concatenate features with previous heatmap
            x_stage = torch.cat([features, stage_outputs[-1]], dim=1)
            heatmap = stage(x_stage)
            stage_outputs.append(heatmap)
        
        # Stack all stages for loss computation
        # Shape: (batch, num_stages, num_landmarks, H, W)
        all_stages = torch.stack(stage_outputs, dim=1)
        
        return all_stages


def soft_argmax_2d(heatmaps, normalize=True):
    """
    Convert heatmaps to (x, y) coordinates using soft-argmax.
    
    Args:
        heatmaps: (B, K, H, W) tensor
        normalize: If True, output in [0, 1], else in pixel coordinates
    
    Returns:
        coords: (B, K, 2) tensor with (x, y) coordinates
    """
    B, K, H, W = heatmaps.shape
    
    # Flatten spatial dimensions
    heatmaps_flat = heatmaps.view(B, K, -1)
    
    # Apply softmax
    heatmaps_flat = F.softmax(heatmaps_flat * 10.0, dim=-1)  # temperature scaling
    
    # Create coordinate grids
    if normalize:
        y_coords = torch.linspace(0, 1, H, device=heatmaps.device)
        x_coords = torch.linspace(0, 1, W, device=heatmaps.device)
    else:
        y_coords = torch.arange(H, dtype=torch.float32, device=heatmaps.device)
        x_coords = torch.arange(W, dtype=torch.float32, device=heatmaps.device)
    
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_grid = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (H*W, 2)
    
    # Weighted sum: (B, K, H*W) @ (H*W, 2) -> (B, K, 2)
    coords = torch.einsum('bkn,nc->bkc', heatmaps_flat, coords_grid)
    
    return coords


def get_model(num_landmarks=55, num_stages=6):
    """
    Create multi-stage heatmap prediction model.
    
    Args:
        num_landmarks: Number of landmarks to predict
        num_stages: Number of refinement stages (default: 6)
    
    Returns:
        model: MultiStageHeatmapModel instance
    """
    return MultiStageHeatmapModel(num_landmarks=num_landmarks, num_stages=num_stages)
