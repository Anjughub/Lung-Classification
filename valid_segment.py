import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time

from utils.data_create import get_loaders_combined
from models.unet import AttU_Net 


def main():
    pretrained_model_path = r"C:\Users\akhsh\Desktop\Fun Projects\LungCancer\models_weights\segemnt_best_model.pth"
    data_dir = r"C:\Users\akhsh\Downloads\lung_data\COVID-19_Radiography_Dataset"
    BATCH_SIZE = 8
    IMAGE_SIZE = (256, 256)
    NUM_SAMPLES_TO_VISUALIZE = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Metrics
    # -------------------------
    def dice_coeff(pred, target, smooth=1e-6):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    def iou_score(pred, target, smooth=1e-6):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        total = pred.sum() + target.sum()
        union = total - intersection
        return (intersection + smooth) / (union + smooth)

    # -------------------------
    # Load Data
    # -------------------------
    _, val_loader = get_loaders_combined(
        base_dir=data_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # -------------------------
    # Load Model
    # -------------------------
    model = AttU_Net().to(device)
    checkpoint = torch.load(pretrained_model_path,weights_only=True, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    # -------------------------
    # Validation Loop
    # -------------------------
    val_loss = 0
    total_dice = 0
    total_iou = 0
    total_pixel_acc = 0
    count = 0
    total_time = 0

    sample_images = []
    sample_preds = []
    sample_masks = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            total_time += (end_time - start_time)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs)
            preds_bin = (preds > 0.5).float()

            # Metrics
            for b in range(images.size(0)):
                total_dice += dice_coeff(preds_bin[b], masks[b]).item()
                total_iou += iou_score(preds_bin[b], masks[b]).item()
                total_pixel_acc += (preds_bin[b] == masks[b]).float().mean().item()
                count += 1

            # Store a few samples for visualization
            if len(sample_images) < NUM_SAMPLES_TO_VISUALIZE:
                sample_images.extend(images.cpu())
                sample_preds.extend(preds_bin.cpu())
                sample_masks.extend(masks.cpu())

    val_loss /= len(val_loader)
    avg_dice = total_dice / count
    avg_iou = total_iou / count
    avg_pixel_acc = total_pixel_acc / count
    avg_time_per_batch = total_time / len(val_loader)
    avg_time_per_image = total_time / count
    

    # -------------------------
    # Print Results
    # -------------------------
    print("\n--- Segmentation Validation Results ---")
    print(f"Loss: {val_loss:.4f}")
    print(f"Dice Coefficient: {avg_dice:.4f}")
    print(f"IoU: {avg_iou:.4f}")
    print(f"Pixel Accuracy: {avg_pixel_acc:.4f}")
    print(f"Avg Inference Time per Batch: {avg_time_per_batch:.6f} s")
    print(f"Avg Inference Time per Image: {avg_time_per_image*1000:.2f} ms")

    # -------------------------
    # Visualization
    # -------------------------
    os.makedirs("segmentation_results", exist_ok=True)

    for i in range(min(NUM_SAMPLES_TO_VISUALIZE, len(sample_images))):
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(sample_images[i].permute(1, 2, 0).numpy())
        axes[0].set_title("Image")
        axes[1].imshow(sample_masks[i][0], cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[2].imshow(sample_preds[i][0], cmap="gray")
        axes[2].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"segmentation_results/sample_{i}.png")
        plt.close()

    print(f"Saved example predictions to 'segmentation_results/'")


if __name__ == "__main__":
    main()