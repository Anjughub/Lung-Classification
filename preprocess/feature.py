import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from tqdm import tqdm
from models.classifier import ConvNeXtClassifier


class FeatureExtractor(nn.Module):
    """Feature extractor using the pre-trained ConvNeXt backbone"""
    def __init__(self, convnext_model):
        super().__init__()
        self.backbone = convnext_model.backbone
        
    def forward(self, x):
        x = self.backbone.features(x)
        x = x.mean([-2, -1])  # Global average pooling
        return x


def extract_features(model_path: str, dataloader, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """Extract features using pre-trained ConvNeXt model"""
    print("Loading pre-trained model...")
    
    # Load the trained model
    model = ConvNeXtClassifier(num_classes=4, input_channels=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(model)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    features_list = []
    labels_list = []

    print("Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            
            # Extract features (768-dimensional from ConvNeXt)
            features = feature_extractor(images)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    
    # Concatenate all features and labels
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return features, labels