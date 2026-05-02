import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class CombinedSegmentationDataset(Dataset):
    def __init__(self, base_dir, transform=None, mask_transform=None,
                 image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_extensions = image_extensions
        
        self.image_mask_pairs = []
        self._collect_data()
        
    def _collect_data(self):
        for class_dir in self.base_dir.iterdir():
            if not class_dir.is_dir():
                continue
            img_dir = class_dir / 'images'
            mask_dir = class_dir / 'masks'
            for ext in self.image_extensions:
                for img_path in img_dir.glob(f'*{ext}'):
                    img_name = img_path.name
                    mask_candidates = [
                        mask_dir / img_name,
                        mask_dir / f"{img_path.stem}_mask{img_path.suffix}",
                        mask_dir / f"{img_path.stem}.png",
                        mask_dir / f"{img_path.stem}_mask.png"
                    ]
                    for mask_path in mask_candidates:
                        if mask_path.exists():
                            self.image_mask_pairs.append((img_path, mask_path))
                            break

        print(f"Found {len(self.image_mask_pairs)} valid image-mask pairs from all classes.")

    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
        
        return image, mask


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None, 
                 image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(list(self.images_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.images_dir.glob(f'*{ext.upper()}')))
        
        self.image_files = sorted(self.image_files)
        self.valid_pairs = []
        for img_path in self.image_files:
            mask_candidates = [
                self.masks_dir / img_path.name,
                self.masks_dir / f"{img_path.stem}_mask{img_path.suffix}",
                self.masks_dir / f"{img_path.stem}.png",
                self.masks_dir / f"{img_path.stem}_mask.png",
            ]
            
            for mask_path in mask_candidates:
                if mask_path.exists():
                    self.valid_pairs.append((img_path, mask_path))
                    break
        
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs")
        
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
        
        return image, mask
    
class MaskedRegionDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256), transform=None, return_class=True):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.return_class = return_class

        self.data = []
        self.class_to_idx = {}

        # Walk through class folders
        for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if not class_dir.is_dir():
                continue
            self.class_to_idx[class_dir.name] = class_idx-1

            images_dir = class_dir / "images"
            masks_dir = class_dir / "masks"

            if not images_dir.exists() or not masks_dir.exists():
                continue

            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    mask_file = masks_dir / img_file.name
                    if mask_file.exists():
                        self.data.append((img_file, mask_file, class_dir.name))

        print(f"Found {len(self.data)} image-mask pairs across {len(self.class_to_idx)} classes.")

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, class_name = self.data[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale for binary mask

        image = self.transform(image)
        mask = self.mask_transform(mask)

        binary_mask = (mask > 0.5).float()
        masked_image = image * binary_mask  # (C, H, W)

        if self.return_class:
            label = self.class_to_idx[class_name]  # Integer class index
            return masked_image, label
        return masked_image


from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class PreMaskedClassificationDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256), transform=None, files=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.samples = []
        self.class_to_idx = {}

        for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if not class_dir.is_dir():
                continue

            self.class_to_idx[class_dir.name] = class_idx
            images_dir = class_dir / "images"
            prd_label_dir = class_dir / "masks"

            for img_path in sorted(images_dir.glob("*.png")):
                prd_path = prd_label_dir / img_path.name
                if prd_path.exists():
                    if files is None or img_path in files:
                        self.samples.append((img_path, prd_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, prd_path, label = self.samples[idx]

        original_img = Image.open(img_path).convert("RGB")
        masked_img = Image.open(prd_path).convert("RGB")

        original_tensor = self.transform(original_img)
        masked_tensor = self.transform(masked_img)

        combined = torch.cat([original_tensor, masked_tensor], dim=0)  # [6, H, W]
        return combined, label
