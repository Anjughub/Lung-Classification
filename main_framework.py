"""
main_framework.py
=================
Full pipeline (as per requirements):

  1. Load segmented (RA-UNetX) lung images
  2. Train all three classifiers:
       - VGG16
       - ResNet50
       - ConvNeXt-Tiny  (proposed)
  3. Evaluate each model:
       - Accuracy, Precision, Recall (Sensitivity), Specificity, F1-score, ROC-AUC
       - Confusion Matrix + ROC Curve
       - Training Curves (Acc vs Epoch, Loss vs Epoch)
  4. Efficiency Metrics (all 3 models):
       - Training time, Inference time, # Parameters, Memory usage
  5. Grad-CAM visualisation (all 3 models)
  6. 5-Fold Cross-Validation (ConvNeXt-Tiny ONLY)

Usage
-----
    python main_framework.py
    
Edit the CONFIG section below to match your paths.
"""

import os, time, psutil
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

# ── Project imports ────────────────────────────────────────────────────────
from models.classifier import ConvNeXtClassifier, VGG16Classifier, ResNet50Classifier
from utils.data_create   import get_train_val_dataloaders, PreMaskedClassificationDataset
from train.train_classifier import train_model
from utils.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    print_metrics_table,
    print_efficiency_table,
    CLASS_NAMES,
)
from utils.gradcam import GradCAM, get_target_layer, visualise_gradcam
from utils.cross_validation import run_5fold_cv


# ==========================================================================
# CONFIG  ── edit these paths
# ==========================================================================
DATASET_PATH = r"C:\Users\akhsh\Downloads\archive (5)\COVID-19_Radiography_Dataset"
OUTPUT_DIR   = "framework_results"

BATCH_SIZE   = 8
IMAGE_SIZE   = (256, 256)
NUM_CLASSES  = 4
NUM_EPOCHS   = 10
LR           = 1e-4
NUM_WORKERS  = 0         # set >0 on Linux with multiple CPU cores
CV_EPOCHS    = 10        # epochs per fold (lighter than full training)
GRADCAM_SAMPLES_PER_CLASS = 2   # how many test images to visualise per class

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================================


def get_num_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6   # millions


def measure_memory_mb(model, device, input_channels=6, image_size=(256, 256)):
    """Rough GPU/CPU memory footprint for one forward pass."""
    dummy = torch.zeros(1, input_channels, *image_size).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy)
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        proc = psutil.Process(os.getpid())
        before = proc.memory_info().rss / (1024 ** 2)
        with torch.no_grad():
            _ = model(dummy)
        after = proc.memory_info().rss / (1024 ** 2)
        mem_mb = max(after - before, 0)
    return mem_mb


def measure_inference_time_ms(model, val_loader, device, n_batches=50):
    """Average per-image inference time in milliseconds."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            if i >= n_batches:
                break
            images = images.to(device)
            start = time.perf_counter()
            _ = model(images)
            end   = time.perf_counter()
            batch_time_ms = (end - start) * 1000
            times.append(batch_time_ms / images.size(0))
    return float(np.mean(times))


def evaluate_model(model, val_loader, device):
    """Run full evaluation and return labels, predictions, probabilities."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def run_gradcam_for_model(model, model_name, val_loader, device, out_dir):
    """Generate Grad-CAM overlays for a few test samples."""
    target_layer = get_target_layer(model, model_name)
    cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    saved = {c: 0 for c in range(NUM_CLASSES)}
    total_needed = GRADCAM_SAMPLES_PER_CLASS * NUM_CLASSES

    for images, labels in val_loader:
        for i in range(images.size(0)):
            lbl = labels[i].item()
            if saved[lbl] >= GRADCAM_SAMPLES_PER_CLASS:
                continue

            inp = images[i].unsqueeze(0).to(device)
            pred_idx  = cam.predict_class(inp)
            heatmap   = cam(inp, class_idx=pred_idx)
            pred_label = CLASS_NAMES[pred_idx]
            true_label = CLASS_NAMES[lbl]

            # Reconstruct an approximate original PIL from tensor
            # (de-normalise the first 3 channels)
            img_t = images[i][:3].clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_t = img_t * std + mean
            img_t = img_t.clamp(0, 1)
            img_pil = transforms.ToPILImage()(img_t)

            fname = f"{model_name.lower().replace(' ', '_')}_true{true_label}_pred{pred_label}_{saved[lbl]}.png"
            visualise_gradcam(img_pil, heatmap, pred_label,
                              save_path=str(out_dir / fname))
            saved[lbl] += 1

        if sum(saved.values()) >= total_needed:
            break

    print(f"Grad-CAM saved {sum(saved.values())} images for {model_name}")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = Path(OUTPUT_DIR)
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # ── Data loaders ──────────────────────────────────────────────────────
    print("Loading data loaders …")
    train_loader, val_loader = get_train_val_dataloaders(
        root_dir=DATASET_PATH,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}\n")

    # ── Model definitions ─────────────────────────────────────────────────
    model_registry = {
        "ResNet50":       ResNet50Classifier(num_classes=NUM_CLASSES),
        "ConvNeXt-Tiny":  ConvNeXtClassifier(num_classes=NUM_CLASSES),
        "VGG16":          VGG16Classifier(num_classes=NUM_CLASSES),
    }

    all_histories   = {}
    all_metrics     = {}
    all_efficiency  = {}

    # ── Train + Evaluate each model ───────────────────────────────────────
    for model_name, model in model_registry.items():
        print("=" * 65)
        print(f"  MODEL: {model_name}")
        print("=" * 65)
        save_path = str(out / f"{model_name.lower().replace(' ', '_')}_best.pth")

        # Train
        history = train_model(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            lr=LR,
            device=DEVICE,
            save_path=save_path,
        )
        all_histories[model_name] = history

        # Reload best weights
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.to(DEVICE).eval()

        # Training curves
        plot_training_curves(history, model_name, save_dir=OUTPUT_DIR)

        # Full evaluation
        y_true, y_pred, y_prob = evaluate_model(model, val_loader, DEVICE)
        metrics = compute_all_metrics(y_true, y_pred, y_prob, NUM_CLASSES)
        all_metrics[model_name] = metrics

        # Confusion matrix
        plot_confusion_matrix(
            y_true, y_pred,
            save_path=str(out / f"{model_name.lower().replace(' ', '_')}_confusion.png"),
        )
        # ROC curve
        plot_roc_curve(
            y_true, y_prob,
            save_path=str(out / f"{model_name.lower().replace(' ', '_')}_roc.png"),
        )

        # Efficiency metrics
        num_params  = get_num_params(model)
        mem_mb      = measure_memory_mb(model, DEVICE)
        infer_ms    = measure_inference_time_ms(model, val_loader, DEVICE)
        train_min   = history.get("training_time_min", 0.0)

        all_efficiency[model_name] = {
            "training_time_min": train_min,
            "inference_time_ms": infer_ms,
            "num_params_M":      num_params,
            "memory_mb":         mem_mb,
        }

        print(f"\n  {model_name} Metrics: Acc={metrics['accuracy']*100:.2f}%  "
              f"F1={metrics['f1_score']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
        print(f"  Efficiency: {train_min:.1f} min  |  {infer_ms:.2f} ms/img  "
              f"|  {num_params:.1f}M params  |  {mem_mb:.0f} MB\n")

        # Grad-CAM
        grad_dir = out / f"gradcam_{model_name.lower().replace(' ', '_')}"
        grad_dir.mkdir(exist_ok=True)
        run_gradcam_for_model(model, model_name, val_loader, DEVICE, grad_dir)

    # ── Summary tables ────────────────────────────────────────────────────
    print_metrics_table(all_metrics)
    print_efficiency_table(all_efficiency)

    # ── 5-Fold CV  (ConvNeXt-Tiny only) ──────────────────────────────────
    print("\n\nRunning 5-Fold Cross-Validation for ConvNeXt-Tiny …")
    full_dataset = PreMaskedClassificationDataset(
        root_dir=DATASET_PATH,
        image_size=IMAGE_SIZE,
    )
    cv_summary = run_5fold_cv(
        full_dataset=full_dataset,
        num_classes=NUM_CLASSES,
        num_epochs=CV_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        num_workers=NUM_WORKERS,
    )

    print("\n✅ Framework complete. All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
