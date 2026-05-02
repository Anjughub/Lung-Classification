from models.unet import AttU_Net
from preprocess.preprocess import preprocess_and_save_masks
import torch
from tqdm import tqdm


def main():
    pretrained_model_path = r"C:\Users\akhsh\Desktop\Fun Projects\LungCancer\models_weights\segemnt_best_model.pth"
    data_dir = r"C:\Users\akhsh\Desktop\Fun Projects\LungCancer\COVID-19_Radiography_Dataset"

    model = AttU_Net()
    checkpoint = torch.load(pretrained_model_path,weights_only=True, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    preprocess_and_save_masks(
    root_dir=data_dir,
    model=model,
    device='cuda'
    )


if __name__ == "__main__":
    main()
