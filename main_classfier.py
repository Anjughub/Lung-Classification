from utils.data_create import get_train_val_dataloaders
from LungClassification.models.classifier import ConvNeXtClassifier
from train.train_classifier import train_model
import os

def main():

    save_dir = "checkpoint_classification"
    data_dir = r"C:\Users\akhsh\Desktop\Fun Projects\LungCancer\COVID-19_Radiography_Dataset"
    print("Data dir : ",data_dir)
    print("Save dir : ",save_dir)

    model = ConvNeXtClassifier(num_classes = 4)
    os.makedirs(save_dir,exist_ok=True)

    print("Loading Train Validation Loaders..")
    train_loader, val_loader = get_train_val_dataloaders(
    root_dir=data_dir,
    batch_size=8,
    val_split=0.2,
    image_size=(256, 256),
    num_workers=0  # use >0 if everything is CPU-safe
    )
    print("Done..")

    train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    lr=1e-4,
    device="cuda",
    save_path=save_dir+"/best_model_mutliclass.pth"
    )

if __name__ == "__main__":
    main()
