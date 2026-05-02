import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


try:
    from dataloader import SegmentationDataset, MaskedRegionDataset

except:
    from utils.dataloader import SegmentationDataset,MaskedRegionDataset

def visualize_batch(dataloader, num_samples=4):

    images, masks = next(iter(dataloader))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(min(num_samples, images.shape[0])):
        img = images[i].clone()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        axes[0, i].imshow(img.permute(1, 2, 0))
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Plot mask
        axes[1, i].imshow(masks[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def check_classification_dataset_info(root_dir, num_samples=3):
    dataset = MaskedRegionDataset(root_dir=root_dir, return_class=True)

    print(f"Total samples found: {len(dataset)}")
    print(f"Classes found: {dataset.class_to_idx}")

    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        class_name = [k for k, v in dataset.class_to_idx.items() if v == label][0]

        print(f"\nSample {i + 1}:")
        print(f" - Image shape: {img.shape}")  # [C, H, W]
        print(f" - Class label: {label} ({class_name})")
        print(f" - Non-zero pixels (white region): {img.sum().item():.2f}")



def check_dataset_info(images_dir, masks_dir):
    dataset = SegmentationDataset(images_dir, masks_dir)
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        img, mask = dataset[0]
        print(f"Sample image type: {type(img)}")
        print(f"Sample mask type: {type(mask)}")
        
        if hasattr(img, 'size'):
            print(f"Sample image size: {img.size}")
        if hasattr(mask, 'size'):
            print(f"Sample mask size: {mask.size}")