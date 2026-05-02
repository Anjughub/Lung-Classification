import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import random


try:
    from dataloader import *
    from tools import check_dataset_info
    from preprocess.preprocess import get_transforms
except:
    from utils.dataloader import *
    from utils.tools import check_dataset_info
    from utils.preprocess import get_transforms


def get_train_val_dataloaders(root_dir, batch_size=16, val_split=0.2, image_size=(256, 256), num_workers=4):
    all_image_paths = []

    # Collect all image paths
    root = Path(root_dir)
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        images_dir = class_dir / "images"
        for img_path in sorted(images_dir.glob("*.png")):
            all_image_paths.append(img_path)

    # Shuffle and split
    random.shuffle(all_image_paths)
    split_idx = int(len(all_image_paths) * (1 - val_split))
    train_files = all_image_paths[:split_idx]
    val_files = all_image_paths[split_idx:]

    # Create datasets
    train_dataset = PreMaskedClassificationDataset(root_dir, image_size=image_size, files=train_files)
    val_dataset = PreMaskedClassificationDataset(root_dir, image_size=image_size, files=val_files)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def create_dataloaders(images_dir, masks_dir, batch_size=8, image_size=(256, 256), 
                      train_split=0.8, augment_train=True, num_workers=4):

    train_img_transform, train_mask_transform = get_transforms(
        image_size=image_size, augment=augment_train
    )
    val_img_transform, val_mask_transform = get_transforms(
        image_size=image_size, augment=False
    )

    check_dataset_info(images_dir, masks_dir)

    full_dataset = SegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=train_img_transform,
        mask_transform=train_mask_transform
    )

    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    val_dataset.dataset.transform = val_img_transform
    val_dataset.dataset.mask_transform = val_mask_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

from torch.utils.data import random_split, DataLoader
from torchvision import transforms

def get_loaders_combined(base_dir, image_size=(256, 256), batch_size=16, val_split=0.2):
    common_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    dataset = CombinedSegmentationDataset(
        base_dir=base_dir,
        transform=common_transform,
        mask_transform=common_transform
    )
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def class_create_dataloaders(root_dir, batch_size=8, image_size=(256, 256), 
                                      train_split=0.8, augment_train=True, num_workers=4):

    train_transform, _ = get_transforms(image_size=image_size, augment=augment_train)
    val_transform, _ = get_transforms(image_size=image_size, augment=False)

    # Full dataset
    full_dataset = MaskedRegionDataset(
        root_dir=root_dir,
        image_size=image_size,
        transform=train_transform,
        return_class=True
    )

    # Split into train/val
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Overwrite transform for val dataset
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
