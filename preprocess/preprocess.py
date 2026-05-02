import os
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def preprocess_and_save_masks(root_dir, model=None, image_size=(256, 256), device='cuda'):
    """
    Preprocess images by either:
      1. Using a segmentation model (if model is not None), or
      2. Using pre-existing masks in a 'masks' folder (if model is None).
    
    Saves masked images in 'prd_label' inside each class folder.
    """
    root_dir = Path(root_dir)
    total = 0

    if model is not None:
        print("⚡ Running in MODEL mode")
        model.eval().to(device)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

        for class_dir in sorted(root_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            images_dir = class_dir / "images"
            prd_label_dir = class_dir / "prd_label"
            prd_label_dir.mkdir(exist_ok=True)

            for img_path in tqdm(sorted(images_dir.glob("*.png")), desc=f"{class_dir.name}"):
                out_path = prd_label_dir / img_path.name
                if out_path.exists():
                    continue

                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)

                output = model(img_tensor)[0].squeeze(0)  # [H, W] mask
                mask = (output > 0).float()

                # Repeat to match image shape
                binary_mask = mask.unsqueeze(0).repeat(3, 1, 1)
                masked_tensor = img_tensor[0] * binary_mask

                # Unnormalize and save
                unnormalized = (masked_tensor * std + mean).clamp(0, 1)
                masked_img = TF.to_pil_image(unnormalized.cpu())
                masked_img.save(out_path)

                total += 1

    else:
        print(" Running in MASK mode (using pre-existing masks)")
        for class_dir in sorted(root_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            images_dir = class_dir / "images"
            masks_dir = class_dir / "masks"
            prd_label_dir = class_dir / "prd_label"
            prd_label_dir.mkdir(exist_ok=True)

            for img_path in tqdm(sorted(images_dir.glob("*.png")), desc=f"{class_dir.name}"):
                out_path = prd_label_dir / img_path.name
                if out_path.exists():
                    continue

                mask_path = masks_dir / img_path.name
                if not mask_path.exists():
                    print(f"Mask not found for {img_path}, skipping..")
                    continue

                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L").resize(img.size)

                # Convert mask to binary tensor
                mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
                mask_tensor = (mask_tensor > 0.5).float()

                # Apply mask (3 channels)
                img_tensor = TF.to_tensor(img)
                binary_mask = mask_tensor.unsqueeze(0).repeat(3, 1, 1)
                masked_tensor = img_tensor * binary_mask

                masked_img = TF.to_pil_image(masked_tensor)
                masked_img.save(out_path)

                total += 1

    print(f"\n✅ Preprocessing complete. Total saved: {total}")
