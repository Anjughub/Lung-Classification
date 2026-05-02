import os
import torch

from utils.data_create import create_dataloaders, get_loaders_combined
from models.unet import AttU_Net
from train.train_unet import train_segmentation_model

def main():
    # Set dataset paths
    images_dir = r"C:\Users\akhsh\Downloads\lung_data\COVID-19_Radiography_Dataset"
    epochs = 20
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = get_loaders_combined(
    base_dir=images_dir,
    image_size=(256, 256),
    batch_size=8
)

    # Initialize model
    print("Initializing Attention U-Net model...")
    model = AttU_Net()

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    print("Starting training...")
    trainer, history = train_segmentation_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        learning_rate=1e-3,
        device=device
    )

    print("Training completed!")

if __name__ == "__main__":
    main()
