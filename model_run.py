import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import time

# -------------------------
# Imports for your models
# -------------------------
from models.unet import AttU_Net
from models.classifier import ConvNeXtClassifier

# -------------------------
# Config
# -------------------------
SEG_MODEL_PATH = r"C:\Users\akhsh\Desktop\Fun Projects\LungCancer\models_weights\segemnt_best_model.pth"
CLS_MODEL_PATH = r"C:\Users\akhsh\Desktop\Fun Projects\LungCancer\project\best_model_mutliclass.pth"

INPUT_PATH = r"C:\Users\akhsh\Downloads\lung_data\COVID-19_Radiography_Dataset\COVID\images\COVID-97.png"  # single image or folder
OUTPUT_DIR = "inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (256, 256)
CLASS_NAMES = ['COVID','Lung_Opacity','Normal','Viral Pneumonia']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

# -------------------------
# Load Models
# -------------------------
# Segmentation model
seg_model = AttU_Net().to(device)
checkpoint = torch.load(SEG_MODEL_PATH,weights_only=True, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
seg_model.load_state_dict(checkpoint['model_state_dict'])
seg_model.eval()

# Classification model
cls_model = ConvNeXtClassifier(num_classes=len(CLASS_NAMES)).to(device)
cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=device))
cls_model.eval()

# -------------------------
# Prediction Pipeline
# -------------------------
def process_image(img_pil):
    # Convert to tensor
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # --- Segmentation ---
    with torch.no_grad():
        seg_output = torch.sigmoid(seg_model(img_tensor))
        seg_mask = (seg_output > 0.5).float()

    # Convert mask to 3-channel
    seg_mask_3c = seg_mask.repeat(1, 3, 1, 1)  # [1,3,H,W]
    masked_image = img_tensor * seg_mask_3c 

    # --- Combine as 6 channels ---
    combined_input = torch.cat([img_tensor, masked_image], dim=1)  # [1,6,H,W]

    # --- Classification ---
    with torch.no_grad():
        logits = cls_model(combined_input)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_class_idx]
        pred_prob = probs[pred_class_idx]

    return seg_mask.squeeze(0).cpu().numpy(), pred_class, pred_prob, probs

# -------------------------
# Visualization
# -------------------------
def overlay_mask_on_image(img_pil, mask, alpha=0.5):
    img_np = np.array(img_pil.resize(IMAGE_SIZE))
    mask_resized = cv2.resize((mask[0] * 255).astype(np.uint8), IMAGE_SIZE)
    color_mask = np.zeros_like(img_np)
    color_mask[:, :, 1] = mask_resized  # green overlay
    overlay = cv2.addWeighted(img_np, 1, color_mask, alpha, 0)
    return overlay

# -------------------------
# Run Inference
# -------------------------
if os.path.isdir(INPUT_PATH):
    image_paths = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]
else:
    image_paths = [INPUT_PATH]

times = []

for img_path in tqdm(image_paths, desc="Processing images"):
    img_pil = load_image(img_path)

    # --- Measure inference time ---
    start_time = time.time()
    seg_mask, pred_class, pred_prob, probs = process_image(img_pil)
    end_time = time.time()

    inference_time = end_time - start_time
    times.append(inference_time)

    overlay_img = overlay_mask_on_image(img_pil, seg_mask)

    # Put text with prediction + time
    text = f"{pred_class} ({pred_prob*100:.2f}%) | {inference_time*1000:.1f} ms"
    cv2.putText(overlay_img, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Save image
    out_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

# -------------------------
# Summary of inference speed
# -------------------------
avg_time = np.mean(times)
print(f"\nResults saved in: {OUTPUT_DIR}")
print(f"Average inference time per image: {avg_time*1000:.2f} ms")
print(f"Fastest: {np.min(times)*1000:.2f} ms, Slowest: {np.max(times)*1000:.2f} ms")